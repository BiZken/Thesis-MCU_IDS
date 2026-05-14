"""
This script sends samples to the MCUs over serial and gathers metrics
Usage: python scripts/serial_test.py
"""

import serial
import serial.tools.list_ports
import csv
import time
import os
import statistics
import datetime
import pathlib
import argparse

# put this script in the scripts directory and give it a testset
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_PATH = os.path.join(PROJECT_DIR, "scripts", "test_samples.csv")

CLASS_NAMES = ["Normal", "DDoS", "Reconnaissance", "Web Attack", "Malware/Access"]

BAUD = 115200

# i dont think this is used
MODEL_ABBREV = {
    "Decision Tree Grouped": "DT",
    "XGBoost Grouped": "XGB",
    "Random Forest Grouped": "RF",
    "Naive Bayes Grouped": "NB",
    "ANN Grouped": "ANN",
    "CNN Grouped": "CNN",
    "CNN CBAM Grouped": "CBAM",
    "CNN SE Grouped": "CNN-SE",
    "GRU": "GRU",
    "ProtoNN": "ProtoNN",
    "ProtoNN Bigger": "ProtoNN-BIG",
    "DS-CNN": "DS-CNN",
    "Extra Trees Grouped": "ET",
    "Extra Trees Bigger": "ET-BIG",
    "LightGBM Grouped": "LGB",
    "LightGBM Bigger": "LGB-BIG",
    "DS-CNN CBAM": "DS-CBAM",
    "DS-CNN CBAM v2": "DS-CBAM-V2",
}


# it works on arch
def find_port():
    candidates = sorted(serial.tools.list_ports.comports(), key=lambda p: p.device)
    for p in candidates:
        if "/dev/ttyACM" in p.device or "/dev/ttyUSB" in p.device:
            return p.device
    return "/dev/ttyACM0"  # fallback


def main():
    parser = argparse.ArgumentParser(description="Run inference benchmark over serial.")
    parser.add_argument(
        "port",
        nargs="?",
        default=None,
        help="Serial port (auto-detected if left blank)",
    )
    args = parser.parse_args()

    port = args.port if args.port else find_port()

    # load test samples
    with open(TEST_DATA_PATH) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    n_features = len(header) - 1  # last column is 'label'
    print(f"Loaded {len(rows)} test samples with {n_features} features")

    # Open serial connection
    print(f"Connecting to {port} at {BAUD} baud...")
    ser = serial.Serial(port, BAUD, timeout=1)
    ser.reset_input_buffer()

    # Wait for READY + model name from boot messages (TODO: the model name part does not work lol)
    # give it 30 seconds to boot (overkill)
    model_name = "Unknown"
    found_ready = False
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            line = ser.readline().decode("utf-8", errors="replace").strip()
        except serial.SerialException:
            time.sleep(0.2)
            continue
        if not line:
            continue
        print(f"  Device: {line}")
        if line == "READY":
            found_ready = True
        elif found_ready and ":" in line:
            candidate = line.split(":")[0].strip()
            if candidate in MODEL_ABBREV:
                model_name = candidate
                break
            else:
                found_ready = False

    ser.timeout = 5  # restore longer timeout for inference responses
    abbrev = MODEL_ABBREV.get(model_name, "UNK")
    print(f"  Model : {model_name} ({abbrev})")

    correct = 0
    total = 0
    results_by_class = {}
    conf_matrix = {}  # conf_matrix[true_name][pred_name] = count
    inference_times = []
    sample_log = []  # (index, true_name, pred_name, time_us, status)

    for row in rows:
        features = row[:n_features]
        true_label = int(row[n_features])
        true_name = (
            CLASS_NAMES[true_label]
            if true_label < len(CLASS_NAMES)
            else f"Class_{true_label}"
        )

        payload = ",".join(features) + "\n"
        ser.write(payload.encode("utf-8"))

        response = ser.readline().decode("utf-8", errors="replace").strip()

        if response.startswith("PRED:"):
            parts = response.split(":")
            pred_class = int(parts[1])
            pred_name = parts[2]
            inference_time = parts[3]

            time_us = int(inference_time.rstrip("us"))
            inference_times.append(time_us)

            match = pred_class == true_label
            if match:
                correct += 1
            total += 1

            status = "OK" if match else "MISS"
            ser.write((status + "\n").encode("utf-8"))
            print(
                f"  [{status}] true={true_name:20s} pred={pred_name:20s} ({inference_time})"
            )

            if true_name not in results_by_class:
                results_by_class[true_name] = {"correct": 0, "total": 0, "fp": 0}
            results_by_class[true_name]["total"] += 1
            if match:
                results_by_class[true_name]["correct"] += 1
            else:
                if pred_name not in results_by_class:
                    results_by_class[pred_name] = {"correct": 0, "total": 0, "fp": 0}
                results_by_class[pred_name]["fp"] += 1

            conf_matrix.setdefault(true_name, {})
            conf_matrix[true_name][pred_name] = (
                conf_matrix[true_name].get(pred_name, 0) + 1
            )

            sample_log.append((total, true_name, pred_name, time_us, status))
        else:
            print(f"  [ERR] unexpected response: {response}")
            total += 1

    ser.close()

    # inference time stats
    if inference_times:
        mean_t = statistics.mean(inference_times)
        min_t = min(inference_times)
        max_t = max(inference_times)
        stdev_t = statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0
    else:
        mean_t = min_t = max_t = stdev_t = 0.0

    tested_at = datetime.datetime.now()

    # per class precision, recall, F2
    per_class_metrics = {}
    for cls in CLASS_NAMES:
        st = results_by_class.get(cls, {"correct": 0, "total": 0, "fp": 0})
        tp = st["correct"]
        fn = st["total"] - tp
        fp = st["fp"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
        per_class_metrics[cls] = (prec, rec, f2)

    macro_prec = sum(v[0] for v in per_class_metrics.values()) / len(CLASS_NAMES)
    macro_rec = sum(v[1] for v in per_class_metrics.values()) / len(CLASS_NAMES)
    macro_f2 = sum(v[2] for v in per_class_metrics.values()) / len(CLASS_NAMES)

    # "final" report
    sep = "=" * 60
    dash = "-" * 60
    lines = []
    lines.append(sep)
    lines.append(
        f"Model  : {model_name} ({len(CLASS_NAMES)} classes, {n_features} features)"
    )
    lines.append(f"Tested : {tested_at.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Port   : {port}  |  Baud: {BAUD}")
    lines.append(f"Samples: {total}")
    lines.append(sep)
    lines.append("ACCURACY")
    lines.append(f"  Overall : {correct}/{total}  ({100 * correct / total:.1f}%)")
    lines.append("")
    lines.append("  Per-class breakdown:")
    for cls_name, st in sorted(results_by_class.items()):
        acc = 100 * st["correct"] / st["total"] if st["total"] > 0 else 0
        lines.append(
            f"    {cls_name:20s}: {st['correct']:4d}/{st['total']:<4d}  ({acc:5.1f}%)"
        )
    lines.append(dash)
    lines.append("INFERENCE TIME  (microseconds)")
    lines.append(f"  Mean   : {mean_t:6.1f} us")
    lines.append(f"  Std Dev: {stdev_t:6.1f} us")
    lines.append(f"  Min    : {min_t:6d} us")
    lines.append(f"  Max    : {max_t:6d} us")
    lines.append(f"  Samples: {len(inference_times)}")
    lines.append(dash)
    lines.append(
        f"PRECISION / RECALL / F2  (macro avg: P={macro_prec:.3f}  R={macro_rec:.3f}  F2={macro_f2:.3f})"
    )
    lines.append(f"  {'Class':<20s}  {'Prec':>6}  {'Rec':>6}  {'F2':>6}")
    lines.append("  " + "-" * 43)
    for cls in CLASS_NAMES:
        prec, rec, f2 = per_class_metrics[cls]
        lines.append(f"  {cls:<20s}: {prec:6.3f}  {rec:6.3f}  {f2:6.3f}")
    lines.append("  " + "-" * 43)
    lines.append(
        f"  {'Macro avg':<20s}: {macro_prec:6.3f}  {macro_rec:6.3f}  {macro_f2:6.3f}"
    )
    lines.append(dash)
    col_w = 8
    abbrev_cols = [c[: col_w - 1] for c in CLASS_NAMES]
    header_row = " " * 22 + "".join(f"{a:>{col_w}}" for a in abbrev_cols)
    lines.append("CONFUSION MATRIX  (rows=true, cols=predicted)")
    lines.append(header_row)
    for cls in CLASS_NAMES:
        row_counts = conf_matrix.get(cls, {})
        row_str = f"  {cls:<20s}:"
        for pred_cls in CLASS_NAMES:
            cnt = row_counts.get(pred_cls, 0)
            row_str += f"{cnt:>{col_w}}"
        lines.append(row_str)
    lines.append(sep)
    # ugly ahh code
    report = "\n".join(lines)

    print()
    print(report)

    # save to results/{MODEL}_YYYYMMDD_HHMMSS.txt
    results_dir = pathlib.Path(PROJECT_DIR) / "results"
    results_dir.mkdir(exist_ok=True)
    filename = abbrev + "_" + tested_at.strftime("%Y%m%d_%H%M%S") + ".txt"
    out_path = results_dir / filename

    with open(out_path, "w") as f:
        f.write("PER-SAMPLE LOG\n")
        f.write(
            f"{'#':>5}  {'True Label':20s}  {'Predicted':20s}  {'Time(us)':>8}  Status\n"
        )
        f.write(dash + "\n")
        for idx, true_n, pred_n, t_us, st in sample_log:
            f.write(f"{idx:5d}  {true_n:20s}  {pred_n:20s}  {t_us:8d}  {st}\n")
        f.write("\n")
        f.write(report + "\n")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
