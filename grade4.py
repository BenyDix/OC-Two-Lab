import multiprocessing as mp
import time
import sys
import random
import csv
import queue
import traceback
import matplotlib.pyplot as plt


# ============================================================
# 1. ГЕНЕРАЦИЯ МАТРИЦ В ОТДЕЛЬНОМ ПРОЦЕССЕ (IPC)
# ============================================================

def _generator_worker(N, K, result_queue):
    """Создаёт нижнетреугольную A и обычную B, передаёт их через Queue."""
    try:
        A = [
            [random.randint(0, 9) for _ in range(i + 1)]
            for i in range(N)
        ]
        B = [
            [random.randint(0, 9) for _ in range(K)]
            for _ in range(N)
        ]
        result_queue.put(("ok", A))
        result_queue.put(("ok", B))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))


def _get_generator_message(result_queue, process):
    while True:
        try:
            return result_queue.get(timeout=0.5)
        except queue.Empty:
            if not process.is_alive():
                process.join()
                raise RuntimeError(
                    "Генератор завершился до передачи всех данных. "
                    f"Код завершения: {process.exitcode}"
                )


def generate_matrices_ipc(N, K):
    result_queue = mp.Queue()
    process = mp.Process(
        target=_generator_worker,
        args=(N, K, result_queue)
    )
    process.start()

    try:
        message_a = _get_generator_message(result_queue, process)
        if message_a[0] == "error":
            raise RuntimeError("Ошибка генерации:\n" + message_a[1])

        message_b = _get_generator_message(result_queue, process)
        if message_b[0] == "error":
            raise RuntimeError("Ошибка генерации:\n" + message_b[1])

        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"Генератор завершился с кодом {process.exitcode}."
            )
        return message_a[1], message_b[1]

    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        result_queue.close()
        result_queue.join_thread()


# ============================================================
# 2. УМНОЖЕНИЕ НИЖНЕТРЕУГОЛЬНОЙ МАТРИЦЫ
# ============================================================

def multiply_row(row_index, row, B):
    """C[i][j] = sum(t=0..i) A[i][t] * B[t][j]."""
    columns = len(B[0])
    result = [0] * columns

    # Порядок i -> t -> j: последовательное чтение строк B и C.
    for t in range(len(row)):
        a = row[t]
        brow = B[t]
        for j in range(columns):
            result[j] += a * brow[j]

    return row_index, result


def multiply_single(A, B):
    C = [None] * len(A)
    for i, row in enumerate(A):
        idx, values = multiply_row(i, row, B)
        C[idx] = values
    return C


# ============================================================
# 3. РАБОЧИЕ ПРОЦЕССЫ: STATIC И DYNAMIC
# ============================================================

def _worker_static(start, end, A, B, result_queue, progress_counter):
    try:
        for i in range(start, end):
            idx, values = multiply_row(i, A[i], B)
            with progress_counter.get_lock():
                progress_counter.value += 1
            result_queue.put(("row", idx, values))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))


def _worker_dynamic(next_row_idx, N, A, B, result_queue, progress_counter):
    try:
        while True:
            # Lock защищает выдачу уникального номера строки.
            with next_row_idx.get_lock():
                i = next_row_idx.value
                if i >= N:
                    break
                next_row_idx.value += 1

            idx, values = multiply_row(i, A[i], B)
            with progress_counter.get_lock():
                progress_counter.value += 1
            result_queue.put(("row", idx, values))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))


def multiply_parallel(A, B, T, mode="static", show_progress=True):
    N = len(A)
    if T <= 0:
        raise ValueError("T должно быть больше нуля.")
    if mode not in ("static", "dynamic"):
        raise ValueError("Режим должен быть static или dynamic.")

    T = min(T, N)
    result_queue = mp.Queue()
    progress_counter = mp.Value("i", 0)
    workers = []

    try:
        if mode == "static":
            base, remainder = divmod(N, T)
            start = 0
            for worker_index in range(T):
                count = base + (1 if worker_index < remainder else 0)
                end = start + count
                process = mp.Process(
                    target=_worker_static,
                    args=(start, end, A, B, result_queue, progress_counter)
                )
                process.start()
                workers.append(process)
                start = end
        else:
            next_row_idx = mp.Value("i", 0)
            for _ in range(T):
                process = mp.Process(
                    target=_worker_dynamic,
                    args=(next_row_idx, N, A, B,
                          result_queue, progress_counter)
                )
                process.start()
                workers.append(process)

        C = [None] * N
        received = 0
        last_percent = -1

        while received < N:
            try:
                message = result_queue.get(timeout=0.5)
            except queue.Empty:
                failed = [
                    p for p in workers
                    if p.exitcode not in (None, 0)
                ]
                if failed:
                    raise RuntimeError(
                        "Рабочий процесс завершился с ошибкой: "
                        + ", ".join(str(p.exitcode) for p in failed)
                    )
                if all(not p.is_alive() for p in workers):
                    raise RuntimeError(
                        f"Получено только {received} строк из {N}."
                    )
                continue

            if message[0] == "error":
                raise RuntimeError("Ошибка рабочего процесса:\n" + message[1])

            _, idx, values = message
            if C[idx] is not None:
                raise RuntimeError(f"Строка {idx} получена повторно.")
            C[idx] = values
            received += 1

            if show_progress:
                # Общий счётчик завершённых строк читается под той же блокировкой,
                # с которой рабочие процессы увеличивают его значение.
                # Дополнительно ограничиваем прогресс числом строк, уже собранных
                # главным процессом, чтобы 100% не появлялись преждевременно.
                with progress_counter.get_lock():
                    completed = progress_counter.value
                displayed = min(completed, received)
                percent = displayed * 100 // N
                if percent != last_percent:
                    print(f"\rПрогресс: {percent}%", end="", flush=True)
                    last_percent = percent

        for process in workers:
            process.join()
        for process in workers:
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Рабочий процесс завершился с кодом {process.exitcode}."
                )

        if any(row is None for row in C):
            raise RuntimeError("Матрица результата собрана не полностью.")

        if show_progress:
            if last_percent != 100:
                print("\rПрогресс: 100%", end="", flush=True)
            print()
        return C

    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join()
        result_queue.close()
        result_queue.join_thread()


# ============================================================
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И ГРАФИКИ
# ============================================================

def checksum(C):
    return sum(sum(row) for row in C)


def trimmed_mean(values):
    if len(values) != 5:
        raise ValueError("Требуется ровно 5 измерений.")
    ordered = sorted(values)
    return sum(ordered[1:-1]) / 3


def plot_grade4(results, ipc_results):
    sizes = sorted({r["size"] for r in results})
    process_counts = sorted({r["T"] for r in results})

    # По одному графику на размер: static и dynamic на одном поле.
    for size in sizes:
        plt.figure(figsize=(10, 6))
        for mode in ("static", "dynamic"):
            data = sorted(
                (r for r in results
                 if r["size"] == size and r["mode"] == mode),
                key=lambda r: r["T"]
            )
            plt.plot([r["T"] for r in data],
                     [r["parallel_time"] for r in data],
                     marker="o", label=mode)
        plt.title(f"Static и dynamic: матрицы {size} × {size}")
        plt.xlabel("Количество процессов")
        plt.ylabel("Время умножения, с")
        plt.xticks(process_counts)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"grade4_static_dynamic_{size}.png", dpi=300)
        plt.close()

    # IPC и последовательное умножение — разные этапы, показаны отдельно.
    plt.figure(figsize=(9, 5))
    x = [r["size"] for r in ipc_results]
    ipc = [r["ipc_time"] for r in ipc_results]
    multiplication = [r["sequential_time"] for r in ipc_results]
    width = 0.36
    positions = list(range(len(x)))
    plt.bar([p - width / 2 for p in positions], ipc,
            width=width, label="Генерация + передача IPC")
    plt.bar([p + width / 2 for p in positions], multiplication,
            width=width, label="Последовательное умножение")
    plt.xticks(positions, [str(size) for size in x])
    plt.title("Время генерации/передачи и умножения")
    plt.xlabel("Размер матриц N × N")
    plt.ylabel("Время, с")
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grade4_ipc_vs_multiplication.png", dpi=300)
    plt.close()

    # Дополнительные графики ускорения и эффективности.
    for metric, ylabel, filename, title in [
        ("speedup", "Ускорение", "grade4_speedup.png",
         "Ускорение для static и dynamic"),
        ("efficiency", "Эффективность, %", "grade4_efficiency.png",
         "Эффективность для static и dynamic"),
    ]:
        for size in sizes:
            plt.figure(figsize=(10, 6))
            for mode in ("static", "dynamic"):
                data = sorted(
                    (r for r in results
                     if r["size"] == size and r["mode"] == mode),
                    key=lambda r: r["T"]
                )
                values = [
                    r[metric] * 100 if metric == "efficiency" else r[metric]
                    for r in data
                ]
                plt.plot([r["T"] for r in data], values,
                         marker="o", label=mode)
            if metric == "speedup":
                plt.plot(process_counts, process_counts, "--",
                         label="Идеальное S = T")
                plt.axvline(6, linestyle=":",
                            label="6 физических / 6 логических")
            plt.title(f"{title}, размер {size} × {size}")
            plt.xlabel("Количество процессов")
            plt.ylabel(ylabel)
            plt.xticks(process_counts)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"grade4_{metric}_{size}.png", dpi=300)
            plt.close()


# ============================================================
# 5. АВТОМАТИЧЕСКОЕ ТЕСТИРОВАНИЕ
# ============================================================

def run_benchmarks():
    sizes = [300, 450, 600, 800]
    process_counts = list(range(1, 13)) + [24, 48]
    repeats = 5
    results = []
    ipc_results = []

    print("=" * 65)
    print("БЕНЧМАРК ОЦЕНКИ 4")
    print(f"Размеры: {sizes}")
    print(f"Количество процессов: {process_counts}")
    print(f"Повторов: {repeats}")
    print("=" * 65)

    for N in sizes:
        K = N
        print(f"\nГенерация матриц {N} × {N} отдельным процессом...")
        ipc_times = []
        A = B = None
        for repetition in range(repeats):
            start = time.perf_counter()
            A, B = generate_matrices_ipc(N, K)
            elapsed = time.perf_counter() - start
            ipc_times.append(elapsed)
            print(f"  Генерация + передача IPC {repetition + 1}/{repeats}: "
                  f"{elapsed:.4f} с")
        ipc_time = trimmed_mean(ipc_times)
        print(f"Среднее IPC (после отбрасывания min/max): "
              f"{ipc_time:.4f} с")

        sequential_times = []
        C_single = None
        for repetition in range(repeats):
            start = time.perf_counter()
            C_single = multiply_single(A, B)
            elapsed = time.perf_counter() - start
            sequential_times.append(elapsed)
            print(f"  Последовательный запуск {repetition + 1}: "
                  f"{elapsed:.4f} с")

        sequential_time = trimmed_mean(sequential_times)
        ipc_results.append({
            "size": N,
            "ipc_time": ipc_time,
            "ipc_runs": ";".join(f"{x:.6f}" for x in ipc_times),
            "sequential_time": sequential_time
        })
        print(f"Среднее последовательное время: {sequential_time:.4f} с")

        for T in process_counts:
            if T > N:
                continue
            for mode in ("static", "dynamic"):
                times = []
                print(f"\nN={N}, T={T}, режим={mode}")
                for repetition in range(repeats):
                    start = time.perf_counter()
                    C_parallel = multiply_parallel(
                        A, B, T, mode=mode, show_progress=False
                    )
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)

                    if C_parallel != C_single:
                        raise RuntimeError(
                            f"Результаты не совпали: N={N}, T={T}, {mode}"
                        )
                    print(f"  Повтор {repetition + 1}/{repeats}: "
                          f"{elapsed:.4f} с — OK")

                parallel_time = trimmed_mean(times)
                speedup = sequential_time / parallel_time
                efficiency = speedup / T
                results.append({
                    "size": N,
                    "T": T,
                    "mode": mode,
                    "ipc_time": ipc_time,
                    "sequential_time": sequential_time,
                    "parallel_time": parallel_time,
                    "speedup": speedup,
                    "efficiency": efficiency
                })
                print(f"  Среднее: {parallel_time:.4f} с | "
                      f"Ускорение: {speedup:.2f}x | "
                      f"Эффективность: {efficiency:.2%}")

    fields = ["size", "T", "mode", "ipc_time", "sequential_time",
              "parallel_time", "speedup", "efficiency"]
    with open("benchmark_grade4.csv", "w", newline="",
              encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    with open("grade4_ipc_results.csv", "w", newline="",
              encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=["size", "ipc_time",
                              "ipc_runs", "sequential_time"]
        )
        writer.writeheader()
        writer.writerows(ipc_results)

    plot_grade4(results, ipc_results)

    print("\nБенчмарк оценки 4 завершён.")
    print("CSV: benchmark_grade4.csv, grade4_ipc_results.csv")
    print("Построены графики static/dynamic, IPC и умножения, "
          "ускорения и эффективности.")


# ============================================================
# 6. ПАРАМЕТРЫ КОМАНДНОЙ СТРОКИ И MAIN
# ============================================================

def parse_params():
    if len(sys.argv) != 5:
        print("Использование: python grade4.py N K T static|dynamic")
        print("Пример: python grade4.py 800 800 6 dynamic")
        sys.exit(1)

    try:
        N = int(sys.argv[1])
        K = int(sys.argv[2])
        T = int(sys.argv[3])
    except ValueError:
        print("Ошибка: N, K и T должны быть целыми числами.")
        sys.exit(1)

    mode = sys.argv[4]
    if min(N, K, T) <= 0:
        print("Ошибка: N, K и T должны быть больше нуля.")
        sys.exit(1)
    if N > 2000 or K > 2000:
        print("Ошибка: размеры не должны превышать 2000.")
        sys.exit(1)
    if mode not in ("static", "dynamic"):
        print("Режим должен быть static или dynamic.")
        sys.exit(1)
    return N, K, T, mode


if __name__ == "__main__":
    mp.freeze_support()

    if len(sys.argv) == 2 and sys.argv[1] == "--benchmark":
        run_benchmarks()
        sys.exit(0)

    N, K, T, mode = parse_params()
    print(f"A: нижнетреугольная матрица {N} × {N}")
    print(f"B: обычная матрица {N} × {K}")
    print(f"Процессов: {T}; распределение: {mode}")

    print("\nГенерация матриц отдельным процессом...")
    start = time.perf_counter()
    A, B = generate_matrices_ipc(N, K)
    ipc_time = time.perf_counter() - start
    print(f"Генерация + передача IPC: {ipc_time:.4f} с")

    print("\nПоследовательное умножение...")
    start = time.perf_counter()
    C_single = multiply_single(A, B)
    sequential_time = time.perf_counter() - start
    print(f"Время: {sequential_time:.4f} с")

    print(f"\nПараллельное умножение ({mode})...")
    start = time.perf_counter()
    C_parallel = multiply_parallel(
        A, B, T, mode=mode, show_progress=True
    )
    parallel_time = time.perf_counter() - start
    print(f"Время: {parallel_time:.4f} с")

    if C_parallel == C_single:
        print("Результаты совпадают.")
        print(f"Контрольная сумма: {checksum(C_parallel)}")
        print(f"Ускорение: {sequential_time / parallel_time:.2f}x")
    else:
        print("ОШИБКА: результаты расходятся.")
        sys.exit(2)
