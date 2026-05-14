# Datasheet power values [mW]
P_NANO33  = 3.6 * 3.0    # 10.8 mW
P_M5STICK = 30.0 * 3.3   # 99.0 mW
P_HUZZAH  = 30.0 * 3.3   # 99.0 mW
TIMES_NANO33 = {
    "Decision Tree":   21.4,
    "Random Forest":  111.1,
    "Extra Trees":    137.6,
    "LightGBM":       691.7,
    "XGBoost":        722.8,
    "ANN":            625.6,
    "Depthwise CNN": 2860.2,
    "Naive Bayes":   6189.3,
    "CNN":         30144.8,
    "CNN with SE": 11246.8,
    "CNN with CBAM": 47124.5,
}

TIMES_M5STICK = {
    "Decision Tree":   16.5,
    "Random Forest":   66.2,
    "Extra Trees":     69.1,
    "Naive Bayes":    372.3,
    "ANN":            520.0,
    "XGBoost":        826.1,
    "LightGBM":      1013.0,
    "Depthwise CNN": 1363.3,
    "CNN":           1818.0,
    "CNN with SE":   4864.5,
    "CNN with CBAM": 9111.3,
}

TIMES_HUZZAH = {
    "Decision Tree":    10.2,
    "Random Forest":    74.1,
    "Extra Trees":      76.3,
    "Naive Bayes":     356.0,
    "ANN":             494.8,
    "XGBoost":         860.8,
    "LightGBM":       1007.7,
    "Depthwise CNN":  1357.7,
    "CNN":            1642.3,
    "CNN with SE":    4705.5,
    "CNN with CBAM": 10047.6,
}


def energy_uj(t_us, P_mW):
    """E [uJ] = t [us] * P [mW] * 1e-3."""
    return t_us * P_mW * 1e-3


def print_table(board, P_mW, times):
    print(f"\n{board}  (P = {P_mW:.2f} mW)")
    print(f"  {'Model':<16s} {'Energy (uJ)':>14s}")
    print("  " + "-" * 32)
    for model, t in times.items():
        print(f"  {model:<16s} {energy_uj(t, P_mW):14.2f}")


if __name__ == "__main__":
    print_table("Arduino Nano 33 BLE Sense", P_NANO33,  TIMES_NANO33)
    print_table("M5StickC PLUS2",            P_M5STICK, TIMES_M5STICK)
    print_table("Adafruit HUZZAH32",         P_HUZZAH,  TIMES_HUZZAH)
