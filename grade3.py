import multiprocessing as mp
import time
import sys
import random
import csv
import queue
import matplotlib.pyplot as plt


# ============================================================
# 1. ГЕНЕРАЦИЯ И УМНОЖЕНИЕ МАТРИЦ
# ============================================================

def generate_matrix(rows, cols):
    return [[random.randint(0, 9) for _ in range(cols)] for _ in range(rows)]


def multiply_row(row_index, row, B, order="itj"):
    """Вычисляет одну строку C. Поддерживаются порядки j-t и t-j."""
    n = len(B[0])
    result = [0] * n

    if order == "ijt":
        # i -> j -> t: для каждого элемента C[i][j] читаем столбец B.
        for j in range(n):
            total = 0
            for t in range(len(row)):
                total += row[t] * B[t][j]
            result[j] = total

    elif order == "itj":
        # i -> t -> j: внутренний цикл проходит по строкам B и C.
        for t in range(len(row)):
            a = row[t]
            brow = B[t]
            for j in range(n):
                result[j] += a * brow[j]
    else:
        raise ValueError("Порядок циклов должен быть ijt или itj.")

    return row_index, result


def multiply_single(A, B, order="itj"):
    C = [None] * len(A)
    for i, row in enumerate(A):
        idx, values = multiply_row(i, row, B, order)
        C[idx] = values
    return C


# ============================================================
# 2. РАБОЧИЙ ПРОЦЕСС И ПАРАЛЛЕЛЬНОЕ УМНОЖЕНИЕ
# ============================================================

def _worker_static(A, B, start, end, result_queue, progress_counter, order):
    try:
        for i in range(start, end):
            idx, values = multiply_row(i, A[i], B, order)
            with progress_counter.get_lock():
                progress_counter.value += 1
            result_queue.put(("row", idx, values))
    except BaseException as exc:
        result_queue.put(("error", repr(exc)))


def multiply_parallel(A, B, T, order="itj", show_progress=True):
    rows = len(A)
    if T <= 0:
        raise ValueError("T должно быть больше нуля.")
    if order not in ("ijt", "itj"):
        raise ValueError("Порядок циклов должен быть ijt или itj.")

    T = min(T, rows)
    result_queue = mp.Queue()
    progress_counter = mp.Value("i", 0)
    workers = []

    try:
        base, remainder = divmod(rows, T)
        start = 0

        for worker_index in range(T):
            count = base + (1 if worker_index < remainder else 0)
            end = start + count
            process = mp.Process(
                target=_worker_static,
                args=(A, B, start, end, result_queue, progress_counter, order)
            )
            process.start()
            workers.append(process)
            start = end

        C = [None] * rows
        received = 0
        last_percent = -1

        while received < rows:
            try:
                message = result_queue.get(timeout=0.5)
            except queue.Empty:
                failed = [p for p in workers if p.exitcode not in (None, 0)]
                if failed:
                    raise RuntimeError(
                        "Рабочий процесс завершился с ошибкой: "
                        + ", ".join(str(p.exitcode) for p in failed)
                    )
                if all(not p.is_alive() for p in workers):
                    raise RuntimeError(
                        f"Получено только {received} строк из {rows}."
                    )
                continue

            if message[0] == "error":
                raise RuntimeError("Ошибка рабочего процесса: " + message[1])

            _, idx, values = message
            if C[idx] is not None:
                raise RuntimeError(f"Строка {idx} получена повторно.")
            C[idx] = values
            received += 1

            # Value считает завершённые рабочими процессами строки под lock.
            # min(completed, received) не позволяет показать 100%, пока
            # главный процесс ещё не получил и не собрал все строки матрицы.
            if show_progress:
                with progress_counter.get_lock():
                    completed = progress_counter.value
                displayed = min(completed, received)
                percent = displayed * 100 // rows
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
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def checksum(C):
    return sum(sum(row) for row in C)


def trimmed_mean(values):
    """Пять измерений: отбросить min и max, усреднить остальные три."""
    if len(values) != 5:
        raise ValueError("Для усечённого среднего требуется ровно 5 измерений.")
    ordered = sorted(values)
    return sum(ordered[1:-1]) / 3


def benchmark_sequential(A, B, order, repeats=5):
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = multiply_single(A, B, order)
        times.append(time.perf_counter() - start)
    return trimmed_mean(times), result, times


# ============================================================
# 4. ГРАФИКИ
# ============================================================

def plot_benchmarks(results):
    sizes = sorted({row["size"] for row in results})
    process_counts = sorted({row["T"] for row in results})

    for metric, ylabel, filename, title in [
        ("parallel_time", "Время, с", "grade3_time.png",
         "Время умножения в зависимости от числа процессов"),
        ("speedup", "Ускорение", "grade3_speedup.png",
         "Ускорение в зависимости от числа процессов"),
        ("efficiency", "Эффективность, %", "grade3_efficiency.png",
         "Эффективность в зависимости от числа процессов"),
    ]:
        plt.figure(figsize=(11, 6))
        for size in sizes:
            data = sorted(
                (r for r in results if r["size"] == size),
                key=lambda r: r["T"]
            )
            values = [
                r[metric] * 100 if metric == "efficiency" else r[metric]
                for r in data
            ]
            plt.plot([r["T"] for r in data], values, marker="o",
                     label=f"{size} × {size}")

        if metric == "speedup":
            plt.plot(process_counts, process_counts, "--",
                     label="Идеальное ускорение S = T")
            plt.axvline(6, linestyle=":",
                        label="6 физических / 6 логических процессоров")

        plt.title(title)
        plt.xlabel("Количество процессов")
        plt.ylabel(ylabel)
        plt.xticks(process_counts)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    print("Графики сохранены: grade3_time.png, grade3_speedup.png, "
          "grade3_efficiency.png")


def plot_loop_orders(data):
    sizes = sorted({row["size"] for row in data})
    plt.figure(figsize=(9, 5))
    for size in sizes:
        rows = [r for r in data if r["size"] == size]
        rows.sort(key=lambda r: r["order"])
        plt.plot([r["order"] for r in rows],
                 [r["time"] for r in rows],
                 marker="o", label=f"{size} × {size}")
    plt.title("Сравнение порядков циклов в последовательном умножении")
    plt.xlabel("Порядок циклов")
    plt.ylabel("Время, с")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("grade3_loop_orders.png", dpi=300)
    plt.close()


# ============================================================
# 5. АВТОМАТИЧЕСКОЕ ТЕСТИРОВАНИЕ
# ============================================================

def run_benchmarks():
    sizes = [300, 450, 600, 800]
    process_counts = list(range(1, 13)) + [24, 48]
    repeats = 5
    benchmark_results = []
    loop_results = []

    print("=" * 65)
    print("БЕНЧМАРК ОЦЕНКИ 3")
    print(f"Размеры: {sizes}")
    print(f"Количество процессов: {process_counts}")
    print(f"Повторов: {repeats}")
    print("Основной порядок циклов: i -> t -> j")
    print("=" * 65)

    for size in sizes:
        print(f"\nМатрицы {size} × {size}: генерация...")
        A = generate_matrix(size, size)
        B = generate_matrix(size, size)

        # Сравнение порядка циклов для пункта 2.4.
        # Для экономии времени сравнение выполняется на самом большом размере.
        if size == max(sizes):
            for order in ("ijt", "itj"):
                _, _, raw_times = benchmark_sequential(A, B, order, repeats)
                avg = trimmed_mean(raw_times)
                loop_results.append({
                    "size": size, "order": order, "time": avg
                })
                print(f"Порядок {order}: {avg:.4f} с "
                      f"(замеры: {[round(x, 4) for x in raw_times]})")

        sequential_time, C_single, seq_raw = benchmark_sequential(
            A, B, "itj", repeats
        )
        print(f"Последовательное время (i -> t -> j): "
              f"{sequential_time:.4f} с")

        for T in process_counts:
            if T > size:
                continue

            times = []
            print(f"\nРазмер {size}, T={T}")
            for repetition in range(repeats):
                start = time.perf_counter()
                C_parallel = multiply_parallel(
                    A, B, T, order="itj", show_progress=False
                )
                elapsed = time.perf_counter() - start

                if C_parallel != C_single:
                    raise RuntimeError(
                        f"Результаты не совпали: size={size}, T={T}"
                    )
                times.append(elapsed)
                print(f"  Повтор {repetition + 1}/{repeats}: "
                      f"{elapsed:.4f} с — OK")

            parallel_time = trimmed_mean(times)
            speedup = sequential_time / parallel_time
            efficiency = speedup / T

            benchmark_results.append({
                "size": size,
                "T": T,
                "mode": "static",
                "sequential_time": sequential_time,
                "parallel_time": parallel_time,
                "speedup": speedup,
                "efficiency": efficiency
            })
            print(f"  Среднее: {parallel_time:.4f} с | "
                  f"Ускорение: {speedup:.2f}x | "
                  f"Эффективность: {efficiency:.2%}")

    with open("benchmark_grade3.csv", "w", newline="",
              encoding="utf-8-sig") as file:
        fields = ["size", "T", "mode", "sequential_time",
                  "parallel_time", "speedup", "efficiency"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(benchmark_results)

    with open("grade3_loop_orders.csv", "w", newline="",
              encoding="utf-8-sig") as file:
        fields = ["size", "order", "time"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(loop_results)

    plot_benchmarks(benchmark_results)
    if loop_results:
        plot_loop_orders(loop_results)

    print("\nБенчмарк оценки 3 завершён.")
    print("CSV: benchmark_grade3.csv, grade3_loop_orders.csv")


# ============================================================
# 6. ПАРАМЕТРЫ КОМАНДНОЙ СТРОКИ И MAIN
# ============================================================

def parse_params(raw):
    if len(raw) != 4:
        print("Использование: python grade3.py M K N T")
        print("Пример: python grade3.py 800 800 800 6")
        return None

    try:
        m, k, n, T = map(int, raw)
    except ValueError:
        print("Ошибка: все аргументы должны быть целыми числами.")
        return None

    if min(m, k, n, T) <= 0:
        print("Ошибка: все значения должны быть больше нуля.")
        return None
    if max(m, k, n) > 2000:
        print("Ошибка: размеры матриц не должны превышать 2000.")
        return None
    return m, k, n, T


if __name__ == "__main__":
    mp.freeze_support()

    if len(sys.argv) == 2 and sys.argv[1] == "--benchmark":
        run_benchmarks()
        sys.exit(0)

    params = parse_params(sys.argv[1:])
    if params is None:
        sys.exit(1)

    m, k, n, T = params
    A = generate_matrix(m, k)
    B = generate_matrix(k, n)

    print("--- Последовательное умножение (i -> t -> j) ---")
    start = time.perf_counter()
    C_single = multiply_single(A, B, "itj")
    sequential_time = time.perf_counter() - start
    print(f"Время: {sequential_time:.4f} с")

    print(f"\n--- Параллельное умножение (T={T}) ---")
    start = time.perf_counter()
    C_parallel = multiply_parallel(A, B, T, "itj", show_progress=True)
    parallel_time = time.perf_counter() - start
    print(f"Время: {parallel_time:.4f} с")

    print(f"Контрольная сумма: {checksum(C_parallel)}")
    if C_parallel == C_single:
        print(f"Результаты совпадают. Ускорение: "
              f"{sequential_time / parallel_time:.2f}x")
    else:
        print("ОШИБКА: результаты расходятся.")
        sys.exit(2)
