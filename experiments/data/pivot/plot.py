import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_experiment(name="random_guadalupe", v_line_cx=None):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 11
    })
    sns.set_palette(sns.color_palette("colorblind"))

    df = pd.read_csv(f"./{name}.csv")
    # df = df[df["method"] != "Ewout van den Berg (stim)"]
    df['method'] = df['method'].replace({'qiskit transpile': 'qiskit',
                                         'ours': 'tableau (ours)',
                                         'ours_temp': 'tableau w/ heat (ours)',
                                         'ours_random': 'tableau w/ random pivot (ours)',
                                         'Bravyi et al. (qiskit)': 'qiskit_tableau'})
    # df = df[df["method"] != "tableau w/ heat (ours)"]
    # df = df[df["method"] != "tableau w/ random pivot (ours)"]
    sns.lineplot(df, x="n_gadgets", y="h", hue="method")
    plt.title("H-Gates")
    plt.xlabel("Number of input Gates")
    plt.ylabel("Number of H-Gates")
    plt.savefig(f"./{name}_h.pdf")
    plt.show()
    plt.clf()

    sns.lineplot(df, x="n_gadgets", y="s", hue="method")
    plt.title("S-Gates")
    plt.xlabel("Number of input gates")
    plt.ylabel("Number of S-Gates")
    plt.savefig(f"./{name}_s.pdf")
    plt.show()
    plt.clf()

    sns.lineplot(df, x="n_gadgets", y="cx", hue="method")
    plt.title("CNOT-Gates")
    plt.xlabel("Number of input Gates")
    plt.ylabel("Number of CNOT-Gates")
    # add a vertical line for v_line_cx if not none
    if v_line_cx is not None:
        plt.axhline(y=v_line_cx, color="black", linestyle="--")
    plt.savefig(f".//{name}_cx.pdf")
    plt.show()
    plt.clf()

    sns.lineplot(df, x="n_gadgets", y="depth", hue="method")
    plt.title("Circuit depth")
    plt.xlabel("Number of input Gates")
    plt.ylabel("Circuit Depth")
    # add a vertical line for v_line_cx if not none
    # if v_line_cx is not None:
    #     plt.axhline(y=v_line_cx, color="black", linestyle="--")
    plt.savefig(f".//{name}_depth.pdf")
    plt.show()
    plt.clf()

    sns.lineplot(df, x="n_gadgets", y="2q_depth", hue="method")
    plt.title("2 Qubit Gate depth")
    plt.xlabel("Number of input Gates")
    plt.ylabel("2 Qubit Gate Depth")
    # add a vertical line for v_line_cx if not none
    # if v_line_cx is not None:
    #     plt.axhline(y=v_line_cx, color="black", linestyle="--")
    plt.savefig(f".//{name}_2qdepth.pdf")
    plt.show()
    plt.clf()
# def plot_experiment(name="random_guadalupe", v_line_cx=None):
#     plt.rcParams.update({
#         "text.usetex": True,
#         "font.family": "sans-serif",
#         "font.size": 11
#     })
#     sns.set_palette(sns.color_palette("colorblind"))

#     df = pd.read_csv(f"./{name}.csv")
#     # df = df[df["method"] != "Ewout van den Berg (stim)"]
#     # df = df.groupby('N_gadgets').groupBy('method').agg(
#     #     func=statistics.mean, axis=['H', 'S', 'CX', 'depth'])
#     print(df.keys)
#     df = pd.pivot_table(df, index=['n_gadgets', 'method'], values=[
#                         'h', 's', 'cx', 'depth'], aggfunc='mean').reset_index()
#     print(df.keys)
#     df['method'] = df['method'].replace({'qiskit transpile': 'qiskit',
#                                          'ours': 'tableau (ours)',
#                                          'Bravyi et al. (qiskit)': 'qiskit_tableau'})
#     print(df.keys)
#     sns.lineplot(df, x="n_gadgets", y="h", hue="method")
#     plt.title("H-Gates")
#     plt.xlabel("Number of input Gates")
#     plt.ylabel("Number of H-Gates")
#     plt.savefig(f"./{name}_h.pdf")
#     plt.show()

#     sns.lineplot(df, x="n_gadgets", y="s", hue="method")
#     plt.title("S-Gates")
#     plt.xlabel("Number of input gates")
#     plt.ylabel("Number of S-Gates")
#     plt.savefig(f"./{name}_s.pdf")
#     plt.show()

#     sns.lineplot(df, x="n_gadgets", y="cx", hue="method")
#     plt.title("CNOT-Gates")
#     plt.xlabel("Number of input Gates")
#     plt.ylabel("Number of CNOT-Gates")
#     # add a vertical line for v_line_cx if not none
#     if v_line_cx is not None:
#         plt.axhline(y=v_line_cx, color="black", linestyle="--")
#     plt.savefig(f"./{name}_cx.pdf")
#     plt.show()


def estimate_routing_overhead(arch_name, n_qubits, conv_gadgets):
    print(
        f"Estimating routing overhead for {arch_name} with {n_qubits} qubits")

    for method in ["ours_temp", "ours_random", "ours"]:
        df = pd.read_csv(f"./random_complete_{n_qubits}.csv")
        df_ = df[df["method"] == method]
        df_ = df_[df_["n_gadgets"] > conv_gadgets]

        best_cx_complete = np.mean(df_["2q_depth"])

        df = pd.read_csv(f"./random_{arch_name}.csv")
        df = df[df["method"] == method]
        df = df[df["n_gadgets"] > conv_gadgets]
        best_cx_quito = np.mean(df["2q_depth"])
        print(
            f"{method}: {100.0 * (best_cx_quito - best_cx_complete) / best_cx_quito:.2f} %")


if __name__ == "__main__":
    # df = pd.read_csv(f"./random_complete_5.csv")
    # df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    # df_ = df_[df_["n_gadgets"] > 75]
    # v_line = np.mean(df_["cx"])

    # plot_experiment(name="random_complete_5", v_line_cx=v_line)
    # plot_experiment(name="random_quito", v_line_cx=v_line)

    # df = pd.read_csv(f"./random_complete_7.csv")
    # df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    # df_ = df_[df_["n_gadgets"] > 110]
    # v_line = np.mean(df_["cx"])

    # plot_experiment(name="random_nairobi", v_line_cx=v_line)
    # plot_experiment(name="random_complete_7", v_line_cx=v_line)

    # df = pd.read_csv(f"./random_complete_16.csv")
    # df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    # df_ = df_[df_["n_gadgets"] > 250]
    # v_line = np.mean(df_["cx"])
    # #
    # plot_experiment(name="random_guadalupe", v_line_cx=v_line)
    # plot_experiment(name="random_complete_16", v_line_cx=v_line)

    # df = pd.read_csv(f"./random_complete_27.csv")
    # df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    # df_ = df_[df_["n_gadgets"] > 500]
    # v_line = np.mean(df_["cx"])

    # plot_experiment(name="random_mumbai", v_line_cx=v_line)
    # plot_experiment(name="random_complete_27", v_line_cx=v_line)

    # df = pd.read_csv(f"./random_complete_65.csv")
    # df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    # df_ = df_[df_["n_gadgets"] > 1250]
    # v_line = np.mean(df_["cx"])

    # # plot_experiment(name="random_ithaca", v_line_cx=v_line)
    # plot_experiment(name="random_complete_65", v_line_cx=v_line)

    # df = pd.read_csv(f"./random_complete_127.csv")
    # df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    # df_ = df_[df_["n_gadgets"] > 3000]
    # v_line = np.mean(df_["cx"])

    # plot_experiment(name="random_brisbane", v_line_cx=v_line)
    # plot_experiment(name="random_complete_127", v_line_cx=v_line)

    estimate_routing_overhead("quito", 5, 75)
    estimate_routing_overhead("nairobi", 7, 110)
    estimate_routing_overhead("guadalupe", 16, 250)
    estimate_routing_overhead("mumbai", 27, 500)
    # estimate_routing_overhead("ithaca", 65, 1250)
    # estimate_routing_overhead("brisbane", 127, 3000)
