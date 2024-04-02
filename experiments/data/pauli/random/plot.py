from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def plot_experiment(name="random_guadalupe", v_line_cx=None, v_line_depth=None):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 11
    })
    sns.set_palette(sns.color_palette("colorblind"))

    df = pd.read_csv(f"./{name}.csv")
    # df = df[df["method"] != "Ewout van den Berg (stim)"]

    df["method"] = df["method"].replace({"pauliopt_ucc": "A-UCCSD-set",
                                        "pauliopt_steiner_nc": "PSGS",
                                         "tket_uccs_pair": "UCCSD-pair",
                                         "tket_uccs_set": "UCCSD-set",
                                         "pauliopt_divide_conquer": "SD\&C", "pauliopt_steiner_clifford": "PSGS-PRC"})

    # df["cx"] = (df["circ_cx"] - df["naive_cx"]) / df["naive_cx"] * 100.0

    # add a vertical line for v_line_cx if not none
    if v_line_cx is not None:
        plt.axhline(y=v_line_depth, color="black", linestyle="--")
    plt.savefig(f".//{name}_2qdepth.pdf")
    plt.show()
    plt.clf()

    sns.lineplot(df, x="n_gadgets", y="cx", hue="method")
    plt.title("CX-Gates")
    plt.xlabel("Number of input Pauli-Polynomials")
    plt.ylabel("Number of H-Gates")
    plt.savefig(f"./{name}_cx.pdf")
    plt.show()
    plt.clf()

    sns.lineplot(df, x="n_gadgets", y="2q_depth", hue="method")
    plt.title("2 Qubit Gate depth")
    plt.xlabel("Number of input Pauli-Polynomials")
    plt.ylabel("2 Qubit Gate Depth")

    # add a vertical line for v_line_cx if not none
    if v_line_depth is not None:
        plt.axhline(y=v_line_depth, color="black", linestyle="--")
    plt.savefig(f".//{name}_2qdepth.pdf")
    plt.show()
    plt.clf()

    sns.lineplot(df, x="n_gadgets", y="depth", hue="method")
    plt.title("Circuit depth")
    plt.xlabel("Number of input Pauli-Polynomials")
    plt.ylabel("Circuit Depth")
    # add a vertical line for v_line_cx if not none
    # if v_line_depth is not None:
    #     plt.axhline(y=v_line_depth, color="black", linestyle="--")
    plt.savefig(f".//{name}_depth.pdf")
    plt.show()
    plt.clf()


def plot_random_pauli_polynomial_experiment():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 11
    })
    sns.set_palette(sns.color_palette("colorblind"))

    large = [100, 200, 300, 500, 1000]
    small = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    df = pd.read_csv("./random_pauli_polynomial.csv")
    df = df[df["topo"] == "complete"]
    # df["cx"] = (df["circ_cx"] - df["circ_cx"]) / df["naive_cx"] * 100.0
    # df["cx"] = 100
    # print(df["method"]["pauliopt_steiner_clifford"])
    df["method"] = df["method"].replace("pauliopt_ucc", "A-UCCSD-set")
    df["method"] = df["method"].replace("pauliopt_steiner_nc", "PSGS")
    df["method"] = df["method"].replace("tket_uccs_pair", "UCCSD-pair")
    df["method"] = df["method"].replace("tket_uccs_set", "UCCSD-set")
    df["method"] = df["method"].replace("pauliopt_divide_conquer", "SD\&C")
    df["method"] = df["method"].replace("pauliopt_steiner_clifford", "PSGS-C")
    for topology in ["complete", "random"]:
        for size in [large, small]:
            for type in [("circ_cx", "CX Count"), ("circ_depth", "Depth")]:
                df_ = df[df["gadgets"].isin(size)]
                # df_ = df_[df_["method"] != "UCCSD-pair"]
                if topology == "complete":
                    df_ = df_[df_["topo"] == "complete"]
                else:
                    df_ = df_[df_["topo"] != "complete"]
                graph_value = type[0]
                graph_name = type[1]
                sns.barplot(x="gadgets", y=graph_value, hue="method",
                            hue_order=[
                                "SD\&C",
                                "UCCSD-pair",
                                "UCCSD-set",
                                "A-UCCSD-set",
                                "PSGS",
                            ],
                            data=df_)
                plt.xlabel("Number of gadgets")
                plt.ylabel(f"Reduction of {graph_name} %")
                plt.legend(title="Algorithm")
                plt.savefig(f"./random_{size}_{topology}_{graph_value}_new.pdf",
                            bbox_inches='tight')
                plt.show()

#     df_ = df[df["gadgets"].isin([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])]
#     df_ = df_[df_["method"] != "UCCSD-pair"]
#     sns.barplot(x="gadgets", y="circ_cx", hue="method",
#                 hue_order=[
#                     "SD\&C",
#                     "UCCSD-pair",
#                     "UCCSD-set",
#                     "A-UCCSD-set",
#                     "PSGS",
#                     "PSGS-C"
#                 ],
#                 data=df_)
#     plt.xlabel("Number of gadgets")
#     plt.ylabel(r"Reduction of Depth [\%]")
#     plt.legend(title="Algorithm")
#     plt.savefig("./random_small_complete_depth.pdf",
#                 bbox_inches='tight')
#     plt.show()


#     df = pd.read_csv("./random_pauli_polynomial.csv")
#     df = df[df["topo"] != "complete"]
#     # df["cx"] = (df["naive_cx"] - df["circ_cx"]) / df["naive_cx"] * 100.0
#     # df["cx"] = 100

#     df["method"] = df["method"].replace("pauliopt_ucc", "A-UCCSD-set")
#     df["method"] = df["method"].replace("pauliopt_steiner_nc", "PSGS")
#     df["method"] = df["method"].replace("tket_uccs_pair", "UCCSD-pair")
#     df["method"] = df["method"].replace("tket_uccs_set", "UCCSD-set")
#     df["method"] = df["method"].replace("pauliopt_divide_conquer", "SD\&C")
#     df["method"] = df["method"].replace("pauliopt_steiner_clifford", "PSGS-C")

#     df_ = df[df["gadgets"].isin([100, 200, 300, 500, 1000])]
#     # df_ = df_[df_["method"] != "UCCSD-pair"]
#     sns.barplot(x="gadgets", y="circ_depth", hue="method",
#                 hue_order=[
#                     "SD\&C",
#                     "UCCSD-pair",
#                     "UCCSD-set",
#                     "A-UCCSD-set",
#                     "PSGS",
#                     "PSGS-C"
#                 ],
#                 data=df_)
#     plt.xlabel("Number of gadgets")
#     plt.ylabel(r"Reduction of CX count [\%]")
#     plt.legend(title="Algorithm")
#     plt.savefig("./random_large_arch_depth.pdf", bbox_inches='tight')
#     plt.show()

#     df_ = df[df["gadgets"].isin([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])]
#     df_ = df_[df_["method"] != "UCCSD-pair"]
#     sns.barplot(x="gadgets", y="circ_cx", hue="method",
#                 hue_order=[
#                     "SD\&C",
#                     "UCCSD-pair",
#                     "UCCSD-set",
#                     "A-UCCSD-set",
#                     "PSGS",
#                     "PSGS-C"
#                 ],
#                 data=df_)
#     plt.xlabel("Number of gadgets")
#     plt.ylabel(r"Reduction of Depth [\%]")
#     plt.legend(title="Algorithm")
#     plt.savefig("./random_small_arch_new.pdf", bbox_inches='tight')
#     plt.show()

# def plot_random_pauli_polynomial_experiment():
#     plt.rcParams.update({
#         "text.usetex": True,
#         "font.family": "sans-serif",
#         "font.size": 11
#     })
#     sns.set_palette(sns.color_palette("colorblind"))

#     df = pd.read_csv("./random_pauli_polynomial.csv")
#     df = df[df["topo"] == "complete"]
#     # df["cx"] = (df["circ_cx"] - df["circ_cx"]) / df["naive_cx"] * 100.0
#     # df["cx"] = 100
#     # print(df["method"]["pauliopt_steiner_clifford"])
#     df["method"] = df["method"].replace("pauliopt_ucc", "A-UCCSD-set")
#     df["method"] = df["method"].replace("pauliopt_steiner_nc", "PSGS")
#     df["method"] = df["method"].replace("tket_uccs_pair", "UCCSD-pair")
#     df["method"] = df["method"].replace("tket_uccs_set", "UCCSD-set")
#     df["method"] = df["method"].replace("pauliopt_divide_conquer", "SD\&C")
#     df["method"] = df["method"].replace("pauliopt_steiner_clifford", "PSGS-C")

#     df_ = df[df["gadgets"].isin([100, 200, 300, 500, 1000])]
#     # df_ = df_[df_["method"] != "UCCSD-pair"]
#     sns.barplot(x="gadgets", y="circ_depth", hue="method",
#                 hue_order=[
#                     "SD\&C",
#                     "UCCSD-pair",
#                     "UCCSD-set",
#                     "A-UCCSD-set",
#                     "PSGS",
#                 ],
#                 data=df_)
#     plt.xlabel("Number of gadgets")
#     plt.ylabel(r"Reduction of CX count [\%]")
#     plt.legend(title="Algorithm")
#     plt.savefig("./random_large_complete_new.pdf",
#                 bbox_inches='tight')
#     plt.show()

#     df_ = df[df["gadgets"].isin([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])]
#     df_ = df_[df_["method"] != "UCCSD-pair"]
#     sns.barplot(x="gadgets", y="circ_depth", hue="method",
#                 hue_order=[
#                     "SD\&C",
#                     "UCCSD-pair",
#                     "UCCSD-set",
#                     "A-UCCSD-set",
#                     "PSGS",
#                     "PSGS-C"
#                 ],
#                 data=df_)
#     plt.xlabel("Number of gadgets")
#     plt.ylabel(r"Reduction of Depth [\%]")
#     plt.legend(title="Algorithm")
#     plt.savefig("./random_small_complete_new.pdf",
#                 bbox_inches='tight')
#     plt.show()
#     df = pd.read_csv("./random_pauli_polynomial.csv")
#     df = df[df["topo"] != "complete"]
#     # df["cx"] = (df["naive_cx"] - df["circ_cx"]) / df["naive_cx"] * 100.0
#     # df["cx"] = 100

#     df["method"] = df["method"].replace("pauliopt_ucc", "A-UCCSD-set")
#     df["method"] = df["method"].replace("pauliopt_steiner_nc", "PSGS")
#     df["method"] = df["method"].replace("tket_uccs_pair", "UCCSD-pair")
#     df["method"] = df["method"].replace("tket_uccs_set", "UCCSD-set")
#     df["method"] = df["method"].replace("pauliopt_divide_conquer", "SD\&C")
#     df["method"] = df["method"].replace("pauliopt_steiner_clifford", "PSGS-C")

#     df_ = df[df["gadgets"].isin([100, 200, 300, 500, 1000])]
#     # df_ = df_[df_["method"] != "UCCSD-pair"]
#     sns.barplot(x="gadgets", y="circ_depth", hue="method",
#                 hue_order=[
#                     "SD\&C",
#                     "UCCSD-pair",
#                     "UCCSD-set",
#                     "A-UCCSD-set",
#                     "PSGS",
#                     "PSGS-C"
#                 ],
#                 data=df_)
#     plt.xlabel("Number of gadgets")
#     plt.ylabel(r"Reduction of CX count [\%]")
#     plt.legend(title="Algorithm")
#     plt.savefig("./random_large_arch_new.pdf", bbox_inches='tight')
#     plt.show()

#     df_ = df[df["gadgets"].isin([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])]
#     df_ = df_[df_["method"] != "UCCSD-pair"]
#     sns.barplot(x="gadgets", y="circ_depth", hue="method",
#                 hue_order=[
#                     "SD\&C",
#                     "UCCSD-pair",
#                     "UCCSD-set",
#                     "A-UCCSD-set",
#                     "PSGS",
#                     "PSGS-C"
#                 ],
#                 data=df_)
#     plt.xlabel("Number of gadgets")
#     plt.ylabel(r"Reduction of Depth [\%]")
#     plt.legend(title="Algorithm")
#     plt.savefig("./random_small_arch_depth.pdf", bbox_inches='tight')
#     plt.show()


if __name__ == '__main__':
    plot_experiment(name="random_quito")
    # plot_experiment(name="random_complete_65", v_line_cx=v_line,
    #                 v_line_depth=v_line_depth)
