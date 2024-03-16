import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_experiment(name="random_guadalupe", v_line_cx=None, v_line_depth=None):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 11
    })
    sns.set_palette(sns.color_palette("colorblind"))

    df = pd.read_csv(f"./{name}.csv")
    # df = df[df["method"] != "Ewout van den Berg (stim)"]
    df['method'] = df['method'].replace({'qiskit transpile': 'Transpile (Qiskit impl.)',
                                         'ours': 'tableau (proposed)',
                                         'ours_temp': 'tableau w/ heat (proposed)',
                                         'ours_random': 'tableau w/ random pivot (proposed)',
                                         'Bravyi et al. (qiskit)': 'Bravyi and Maslov (Qiskit impl.)',
                                         'Ewout van den Berg (stim)': 'van den Berg (Stim impl.)',
                                         'Maslov et al' : 'Maslov and Yang (Qiskit impl.)',
                                         'Duncan et al (pyzx)': 'Duncan et al (pyzx impl.)'
                                         })
    df = df[df["method"] != "tableau w/ heat (proposed)"]
    df = df[df["method"] != "tableau w/ random pivot (proposed)"]
    fig = sns.lineplot(df, x="n_gadgets", y="h", hue="method")
    handles, labels = fig.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, ncol=3, borderaxespad=1.)
    lgd = fig.legend(handles, labels,
                     bbox_to_anchor=(0.95, 1.35), ncol=3, borderaxespad=1.)
    export_legend(lgd)
    # fig.get_legend().remove()
    # plt.title("H-Gates")
    # plt.xlabel("Number of input Gates")
    # plt.ylabel("Number of H-Gates")
    # plt.tight_layout(pad=1.0)
    # plt.savefig(f"./{name}_h.pdf")
    # plt.show()
    # plt.clf()

    # fig = sns.lineplot(df, x="n_gadgets", y="s", hue="method")
    # fig.get_legend().remove()
    # plt.title("S-Gates")
    # plt.xlabel("Number of input gates")
    # plt.ylabel("Number of S-Gates")
    # plt.tight_layout(pad=1.0)
    # plt.savefig(f"./{name}_s.pdf")
    # plt.show()
    # plt.clf()

    # fig = sns.lineplot(df, x="n_gadgets", y="cx", hue="method")
    # fig.get_legend().remove()
    # plt.title("CNOT-Gates")
    # plt.xlabel("Number of input Gates")
    # plt.ylabel("Number of CNOT-Gates")
    # # add a vertical line for v_line_cx if not none
    # if v_line_cx is not None:
    #     plt.axhline(y=v_line_cx, color="black", linestyle="--")
    # plt.tight_layout(pad=1.0)
    # plt.savefig(f".//{name}_cx.pdf")
    # plt.show()
    # plt.clf()

    # fig = sns.lineplot(df, x="n_gadgets", y="depth", hue="method")
    # fig.get_legend().remove()
    # plt.title("Circuit depth")
    # plt.xlabel("Number of input Gates")
    # plt.ylabel("Circuit Depth")
    # # add a vertical line for v_line_cx if not none
    # # if v_line_cx is not None:
    # #     plt.axhline(y=v_line_cx, color="black", linestyle="--")
    # plt.tight_layout(pad=1.0)
    # plt.savefig(f".//{name}_depth.pdf")
    # plt.show()
    # plt.clf()

    # fig = sns.lineplot(df, x="n_gadgets", y="2q_depth", hue="method")
    # fig.get_legend().remove()
    # plt.title("2 Qubit Gate depth")
    # plt.xlabel("Number of input Gates")
    # plt.ylabel("2 Qubit Gate Depth")
    # # add a vertical line for v_line_cx if not none
    # if v_line_depth is not None:
    #     plt.axhline(y=v_line_depth, color="black", linestyle="--")
    # plt.tight_layout(pad=1.0)
    # plt.savefig(f".//{name}_2qdepth.pdf")
    # plt.show()
    # plt.clf()

    # fig = sns.lineplot(df, x="n_gadgets", y="depth", hue="method")
    # fig.get_legend().remove()
    # plt.title("Circuit depth")
    # plt.xlabel("Number of input Gates")
    # plt.ylabel("Circuit Depth")
    # # add a vertical line for v_line_cx if not none
    # # if v_line_depth is not None:
    # #     plt.axhline(y=v_line_depth, color="black", linestyle="--")
    # plt.tight_layout(pad=1.0)
    # plt.savefig(f".//{name}_depth.pdf")
    # plt.show()
    plt.clf()


def plot_full_experiment(data="cx", v_line_cx=None, v_line_depth=None):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 11
    })
    sns.set_palette(sns.color_palette("colorblind"))

    df = pd.read_csv(f"./{name}.csv")
    # df = df[df["method"] != "Ewout van den Berg (stim)"]
    df['method'] = df['method'].replace({'qiskit transpile': 'Transpile (Qiskit impl.)',
                                         'ours': 'tableau (ours)',
                                         'ours_temp': 'tableau w/ heat (ours)',
                                         'ours_random': 'tableau w/ random pivot (ours)',
                                         'Bravyi et al. (qiskit)': 'Bravyi et al. (Qiskit impl.)',
                                         'Ewout van den Berg (stim)': 'Van den Berg (Stim impl.)'
                                         })
    df = df[df["method"] != "tableau w/ heat (ours)"]
    df = df[df["method"] != "tableau w/ random pivot (ours)"]
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
    if v_line_depth is not None:
        plt.axhline(y=v_line_depth, color="black", linestyle="--")
    plt.savefig(f".//{name}_2qdepth.pdf")
    plt.show()
    plt.clf()

    sns.lineplot(df, x="n_gadgets", y="depth", hue="method")
    plt.title("Circuit depth")
    plt.xlabel("Number of input Gates")
    plt.ylabel("Circuit Depth")
    # add a vertical line for v_line_cx if not none
    # if v_line_depth is not None:
    #     plt.axhline(y=v_line_depth, color="black", linestyle="--")
    plt.savefig(f".//{name}_depth.pdf")
    plt.show()
    plt.clf()


def export_legend(legend, filename="legend.pdf", expand=[-5, -5, 5, 5]):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


# def plot_two_experiment(name_restricted="random_guadalupe", name_complete="random_complete_16", v_line_cx=None, v_line_depth=None):
#     plt.rcParams.update({
#         "text.usetex": True,
#         "font.family": "sans-serif",
#         "font.size": 11
#     })
#     sns.set_palette(sns.color_palette("colorblind"))
#     name = name_restricted + '_' + name_complete

#     df_restricted = pd.read_csv(f"./{name_restricted}.csv")
#     df_complete = pd.read_csv(f"./{name_complete}.csv")
#     # df = df[df["method"] != "Ewout van den Berg (stim)"]
#     df_restricted['method'] = df_restricted['method'].replace({'qiskit transpile': 'Transpile (Qiskit impl.)',
#                                                                'ours': 'tableau (ours)',
#                                                                'ours_temp': 'tableau w/ heat (ours)',
#                                                                'ours_random': 'tableau w/ random pivot (ours)',
#                                                                'Bravyi et al. (qiskit)': 'Bravyi et al. (Qiskit impl.)',
#                                                                'Ewout van den Berg (stim)': 'Van den Berg (Stim impl.)'
#                                                                })
#     df_restricted = df_restricted[df_restricted["method"]
#                                   != "tableau w/ heat (ours)"]
#     df_restricted = df_restricted[df_restricted["method"]
#                                   != "tableau w/ random pivot (ours)"]

#     df_complete['method'] = df_complete['method'].replace({'qiskit transpile': 'Transpile (Qiskit impl.)',
#                                                            'ours': 'tableau (ours)',
#                                                            'ours_temp': 'tableau w/ heat (ours)',
#                                                            'ours_random': 'tableau w/ random pivot (ours)',
#                                                            'Bravyi et al. (qiskit)': 'Bravyi et al. (Qiskit impl.)',
#                                                            'Ewout van den Berg (stim)': 'Van den Berg (Stim impl.)'
#                                                            })
#     df_complete = df_complete[df_complete["method"]
#                               != "tableau w/ heat (ours)"]
#     df_complete = df_complete[df_complete["method"]
#                               != "tableau w/ random pivot (ours)"]

    # fig, (ax1, ax2) = plt.subplots(
    #     1, 2, sharex=True, figsize=(9, 4))
    # gfg = sns.lineplot(df_restricted, x="n_gadgets", y="h", hue="method",
    #                    ax=ax1)
    # gfg.set_xlabel("Number of input Gates")
    # gfg.set_ylabel("Number of H Gates")

    # gfg = sns.lineplot(df_complete, x="n_gadgets", y="h", hue="method", ax=ax2)
    # gfg.set_xlabel("Number of input Gates")
    # gfg.set_ylabel("")
    # ax1.get_legend().remove()
    # ax2.get_legend().remove()
    # handles, labels = ax1.get_legend_handles_labels()
    # lgd = fig.legend(handles, labels,
    #                  bbox_to_anchor=(0.95, 1.05), ncol=4, borderaxespad=1.)
    # plt.savefig(f"./{name}_h.pdf", bbox_extra_artists=(lgd,),
    #             bbox_inches='tight')
    # plt.show()
    # plt.clf()

    # fig, (ax1, ax2) = plt.subplots(
    #     1, 2, sharex=True, figsize=(9, 4))
    # gfg = sns.lineplot(df_restricted, x="n_gadgets", y="s", hue="method",
    #                    ax=ax1)
    # gfg.set_xlabel("Number of input Gates")
    # gfg.set_ylabel("Number of S Gates")

    # gfg = sns.lineplot(df_complete, x="n_gadgets", y="s", hue="method", ax=ax2)
    # gfg.set_xlabel("Number of input Gates")
    # gfg.set_ylabel("")
    # ax1.get_legend().remove()
    # ax2.get_legend().remove()
    # handles, labels = ax1.get_legend_handles_labels()
    # lgd = fig.legend(handles, labels,
    #                  bbox_to_anchor=(0.95, 1.05), ncol=4, borderaxespad=1.)
    # plt.savefig(f"./{name}_s.pdf", bbox_extra_artists=(lgd,),
    #             bbox_inches='tight')
    # plt.show()
    # plt.clf()

    # fig, (ax1, ax2) = plt.subplots(
    #     1, 2, sharex=True, figsize=(15, 5))
    # plt.subplots_adjust(wspace=0.3)
    # gfg = sns.lineplot(df_restricted, x="n_gadgets", y="cx", hue="method",
    #                    ax=ax1)
    # gfg.set_title("CNOT-Gates")
    # gfg.set_xlabel("Number of input Gates")
    # gfg.set_ylabel("Number of CNOT Gates")
    # if v_line_cx is not None:
    #     gfg.axhline(y=v_line_cx, color="black", linestyle="--")

    # gfg = sns.lineplot(df_complete, x="n_gadgets",
    #                    y="cx", hue="method", ax=ax2)
    # gfg.set_title("CNOT-Gates")
    # gfg.set_xlabel("Number of input Gates")
    # gfg.set_ylabel("Number of CNOT Gates")
    # ax1.get_legend().remove()
    # ax2.get_legend().remove()
    # if v_line_cx is not None:
    #     gfg.axhline(y=v_line_cx, color="black", linestyle="--")
    # handles, labels = ax1.get_legend_handles_labels()
    # lgd = fig.legend(handles, labels, ncol=3, borderaxespad=1.)
    # # plt.savefig(f"./{name}_cx.pdf", bbox_extra_artists=(lgd,),
    # #             bbox_inches='tight')
    # export_legend(lgd)
    # plt.show()
    # plt.clf()
    # fig, (ax1, ax2) = plt.subplots(ncols=2)
    # sns.lineplot(df_restricted, x="n_gadgets", y="s", hue="method", ax=ax1)
    # sns.lineplot(df_complete, x="n_gadgets", y="s", hue="method", ax=ax2)
    # plt.title("S-Gates")
    # plt.xlabel("Number of input gates")
    # plt.ylabel("Number of S-Gates")
    # plt.savefig(f"./{name}_s.pdf")
    # plt.show()
    # plt.clf()

    # fig, (ax1, ax2) = plt.subplots(ncols=2)
    # sns.lineplot(df_restricted, x="n_gadgets", y="cx", hue="method", ax=ax1)
    # sns.lineplot(df_complete, x="n_gadgets", y="cx", hue="method", ax=ax2)
    # plt.title("CNOT-Gates")
    # plt.xlabel("Number of input Gates")
    # plt.ylabel("Number of CNOT-Gates")
    # # add a vertical line for v_line_cx if not none
    # if v_line_cx is not None:
    #     plt.axhline(y=v_line_cx, color="black", linestyle="--")
    # plt.savefig(f".//{name}_cx.pdf")
    # plt.show()
    # plt.clf()

    # fig, (ax1, ax2) = plt.subplots(ncols=2)
    # sns.lineplot(df_restricted, x="n_gadgets", y="depth", hue="method", ax=ax1)
    # sns.lineplot(df_complete, x="n_gadgets", y="depth", hue="method", ax=ax2)
    # plt.title("Circuit depth")
    # plt.xlabel("Number of input Gates")
    # plt.ylabel("Circuit Depth")
    # # add a vertical line for v_line_cx if not none
    # # if v_line_cx is not None:
    # #     plt.axhline(y=v_line_cx, color="black", linestyle="--")
    # plt.savefig(f".//{name}_depth.pdf")
    # plt.show()
    # plt.clf()

    # fig, (ax1, ax2) = plt.subplots(ncols=2)
    # sns.lineplot(df_restricted, x="n_gadgets",
    #              y="2q_depth", hue="method", ax=ax1)
    # sns.lineplot(df_complete, x="n_gadgets",
    #              y="2q_depth", hue="method", ax=ax2)
    # plt.title("2 Qubit Gate depth")
    # plt.xlabel("Number of input Gates")
    # plt.ylabel("2 Qubit Gate Depth")
    # # add a vertical line for v_line_cx if not none
    # if v_line_depth is not None:
    #     plt.axhline(y=v_line_depth, color="black", linestyle="--")
    # plt.savefig(f".//{name}_2qdepth.pdf")
    # plt.show()
    # plt.clf()


def estimate_routing_overhead(arch_name, n_qubits, conv_gadgets):
    print(
        f"Estimating routing overhead for {arch_name} with {n_qubits} qubits")

    for method in ["Bravyi et al. (qiskit)", "Ewout van den Berg (stim)", "Duncan et al (pyzx)", "Maslov et al", "ours"]:
        df = pd.read_csv(f"./random_complete_{n_qubits}.csv")
        df_ = df[df["method"] == method]
        df_ = df_[df_["n_gadgets"] > conv_gadgets]

        best_cx_complete = np.mean(df_["2q_depth"])

        df = pd.read_csv(f"./random_{arch_name}.csv")
        df = df[df["method"] == method]
        df = df[df["n_gadgets"] > conv_gadgets]
        best_cx_arch = np.mean(df["2q_depth"])
        print(
            f"{method}: {100.0 * (best_cx_arch - best_cx_complete) / best_cx_arch:.2f} %")


if __name__ == "__main__":
    df = pd.read_csv(f"./random_complete_5.csv")
    df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    df_ = df_[df_["n_gadgets"] > 75]
    v_line = np.mean(df_["cx"])
    v_line_depth = np.mean(df_["2q_depth"])

    plot_experiment(name="random_complete_5", v_line_cx=v_line,
                    v_line_depth=v_line_depth)
    # plot_experiment(name="random_quito", v_line_cx=v_line,
    #                 v_line_depth=v_line_depth)

    # df = pd.read_csv(f"./random_complete_7.csv")
    # df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    # df_ = df_[df_["n_gadgets"] > 110]
    # v_line = np.mean(df_["cx"])
    # v_line_depth = np.mean(df_["2q_depth"])

    # plot_experiment(name="random_nairobi", v_line_cx=v_line,
    #                 v_line_depth=v_line_depth)
    # plot_experiment(name="random_complete_7", v_line_cx=v_line,
    #                 v_line_depth=v_line_depth)

    # df = pd.read_csv(f"./random_complete_16.csv")
    # df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    # df_ = df_[df_["n_gadgets"] > 250]
    # v_line = np.mean(df_["cx"])
    # v_line_depth = np.mean(df_["2q_depth"])
    # #
    # plot_experiment(name="random_guadalupe", v_line_cx=v_line,
    #                 v_line_depth=v_line_depth)
    # plot_experiment(name="random_complete_16", v_line_cx=v_line,
    #                 v_line_depth=v_line_depth)

    # df = pd.read_csv(f"./random_complete_27.csv")
    # df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    # df_ = df_[df_["n_gadgets"] > 500]
    # v_line = np.mean(df_["cx"])
    # df_ = df[df["method"] == "Maslov et al"]
    # df_ = df_[df_["n_gadgets"] > 500]
    # v_line_depth = np.mean(df_["2q_depth"])

    # plot_experiment(name="random_mumbai", v_line_cx=v_line,
    #                 v_line_depth=v_line_depth)
    # plot_experiment(name="random_complete_27", v_line_cx=v_line,
    #                 v_line_depth=v_line_depth)

    # df = pd.read_csv(f"./random_complete_65.csv")
    # df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    # df_ = df_[df_["n_gadgets"] > 1250]
    # v_line = np.mean(df_["cx"])
    # df_ = df[df["method"] == "Maslov et al"]
    # df_ = df_[df_["n_gadgets"] > 1250]
    # v_line_depth = np.mean(df_["2q_depth"])

    # plot_experiment(name="random_ithaca", v_line_cx=v_line,
    #                 v_line_depth=v_line_depth)
    # plot_experiment(name="random_complete_65", v_line_cx=v_line,
    #                 v_line_depth=v_line_depth)

    # df = pd.read_csv(f"./random_complete_127.csv")
    # df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    # df_ = df_[df_["n_gadgets"] > 3000]
    # v_line = np.mean(df_["cx"])
    # df_ = df[df["method"] == "Maslov et al"]
    # df_ = df_[df_["n_gadgets"] > 500]
    # v_line_depth = np.mean(df_["2q_depth"])

    # plot_experiment(name="random_brisbane", v_line_cx=v_line,
    #                 v_line_depth=v_line_depth)
    # plot_experiment(name="random_complete_127",
    #                 v_line_cx=v_line, v_line_depth=v_line_depth)

    estimate_routing_overhead("quito", 5, 75)
    estimate_routing_overhead("nairobi", 7, 110)
    estimate_routing_overhead("guadalupe", 16, 250)
    estimate_routing_overhead("mumbai", 27, 500)
    estimate_routing_overhead("ithaca", 65, 1250)
    estimate_routing_overhead("brisbane", 127, 3000)
