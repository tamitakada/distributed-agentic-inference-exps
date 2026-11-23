#!/usr/bin/env python3
import sys
import json
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from scipy import stats


plt.style.use("fivethirtyeight")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

def plot_energy_usage_vs_prompt_len(df: pd.DataFrame):
    df = df[df["status"]=="ok"]
        
    fig, ax = plt.subplots()

    ax.plot(df["n_tokens"].unique(), df.groupby("n_tokens")["energy_J"].mean())

    ax.grid(True)
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Energy consumption (J)")
    ax.set_title("Energy Consumption of Prefill Computation vs. Prompt Length\n Qwen3-Coder-30B-A3B-Instruct on 1xGH200")
    
    plt.tight_layout()
    plt.show()


def plot_compute_transfer_lat_vs_prompt_len(df: pd.DataFrame):
    df = df[df["status"]=="ok"]
        
    fig, ax = plt.subplots()

    ax.plot(df["n_tokens"].unique(), df.groupby("n_tokens")["latency_s"].mean(), label="recompute")
    
    nvlink_bw = 450
    total_gb = df["kv_cache_GB"]
    
    for i in [1, 5, 10]:
        wan_bw = 0.125 * i # GB / s
        transfer_s = total_gb / wan_bw + 2 * (total_gb / nvlink_bw) + 2 * (total_gb / 50)
        ax.plot(df["n_tokens"].unique(), transfer_s.unique(), label=f"transfer ({i} Gbps)")
    
    m, b = np.polyfit(df["n_tokens"].unique(),
                      df.groupby("n_tokens")["kv_cache_GB"].mean(), 1)
    
    def tok_to_kv(toks):
        return toks * m + b
    
    def kv_to_tok(kvs):
        return (kvs - b) / m
    
    axtop = ax.secondary_xaxis("top", functions=(tok_to_kv, kv_to_tok))
    axtop.set_xlabel("KV cache size (GB)")
    
    ax.legend()

    ax.grid(True)
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Prefill and Transfer Latency vs. Prompt Length\n Qwen3-Coder-30B-A3B-Instruct on 1xGH200")
    
    plt.tight_layout()
    plt.show()


def plot_vs_cached_ratio(dfs: list[pd.DataFrame]):
    assert(all((df["status"]=="ok").all() for df in dfs))
    
    for df in dfs:
        print(df["hit_pct_actual"] - df["hit_pct_requested"])
    
    fig, ax = plt.subplots(1, ncols=2, figsize=(8, 5))
    
    for df in sorted(dfs, key=lambda x: x["prompt_tokens"].mean()):
        wan_bw = 1.25 # GB / s
        pcie4_16_bw = 32
        df["transfer_s"] = df["kv_reused_GB"] / wan_bw + 4 * (df["kv_reused_GB"] / pcie4_16_bw)
        df["combined_s"] = df["latency_s"] + df["transfer_s"]
    
        ax[0].plot(df["hit_pct_requested"].unique(),
                   df.groupby("hit_pct_requested")["combined_s"].mean(),
                   label=f"{df['prompt_tokens'].max()} token prompt")
        
        ax[1].plot(df["hit_pct_requested"].unique(),
                   df.groupby("hit_pct_requested")["energy_J"].mean(),
                   label=f"{df['prompt_tokens'].max()} token prompt")
    
    ax[0].legend()
    ax[0].set_xlabel("% cached")
    ax[0].set_ylabel("prefill latency (s)")
    ax[0].set_title("Prefill Latency vs. % Cached Prompt Tokens\n Qwen3-8B on 1xA30")
    ax[0].grid(True)
    
    ax[1].legend()
    ax[1].set_xlabel("% cached")
    ax[1].set_ylabel("total energy consumption (J)")
    ax[1].set_title("Energy Consumption (Compute Only) vs. % Cached Prompt Tokens\n Qwen3-8B on 1xA30")
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_prefill_lat_vs_out_toks(res: dict):
    budgets = sorted(set([r["budget"] if r["budget"] else 0 for r in res]))

    n = len(budgets)
    slot = 0.95 / n   # horizontal space per bar
    w = slot * 0.8   # bar width (leaves a gap between bars)
    
    cmap = plt.colormaps["viridis"]
     
    diffs = ["easy", "medium", "hard"]
    
    for key, keyname in zip(["time_s", "energy_J"], ["latency (s)", "energy consumption (J)"]):
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = [a for axs in axes for a in axs]
        axes[-1].set_axis_off()
        
        for ax, diff in zip(axes, diffs):
            xticks, xlabels = [], []
            problems = sorted(set(r["title"] for r in res if r["diff"]==diff))
            for i, p in enumerate(problems):
                for j, b in enumerate(budgets):
                    budget = None if b == 0 else b
                    data = [r for r in res if r["title"] == p and r["budget"] == budget]
                    if len(data) != 3:
                        print(p, b)
                    
                    assert(len(data) > 0)
                    
                    mean = np.mean([d[key] for d in data])
                    std = np.std([d[key] for d in data])
                    acc = len([d["correct"] for d in data]) / len(data) * 100
                    mean_tok = np.mean([d["n_tok"] for d in data])
                    out_label = f"{int(mean_tok)}" if mean_tok < 1000 else f"{int(mean_tok // 1000)}K"
                    
                    x = i + (j - (n - 1) / 2) * slot
                    
                    ax.bar(x, mean, w, color="tab:green" if acc == 100 else "tab:red")
                    ax.text(x, mean, out_label, ha="center", va="bottom", fontsize=8)
                    xticks.append(x)
                    xlabels.append(str(b))
                ax.text(i, -0.16, f"P{i+1}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11)
                
            ax.set_title(f"Latency vs. output token budget (diff={diff})")
            ax.set_xticks(xticks, xlabels, rotation=45)
            ax.set_ylabel(keyname)
            ax.set_xlabel("budget (# output tokens over bars)", labelpad=10)
#            ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="tab:green"),
#                               plt.Rectangle((0, 0), 1, 1, color="tab:red")],
#                      labels=["correct", "incorrect"], loc="lower right")
        
        plt.tight_layout()
        plt.show()
    

def plot_compactions_cdf(df: pd.DataFrame):
    print((df["status"]!="ok").sum())
    
    df = df[df["status"]=="ok"]

    policies = sorted(df["policy"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for ax, key, keyname in zip(axes, ["latency_s", "gpu_energy_j"],
                                ["compaction latency (s)", "GPU energy usage (J)"]):
        
        for p in policies:
            dfp = df[df["policy"]==p]
            
            params = dfp.iloc[0]["params"]
            threshold = params["max_size"]
            keep = params["keep_first"]
            
            sns.kdeplot(x=dfp.groupby("session_id")[key].mean(), cumulative=True, ax=ax,
                        label=f"threshold={threshold}, keep_fst={keep}", log_scale=True)
        
        
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.set_xlabel(keyname)
        ax.set_ylabel("CDF")
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    

def stats_compaction(df: pd.DataFrame):
    """
        * Is the # summary tokens correlated w/ # forgotten events?
        * Is the # summary tokens correlated w/ # forgotten tokens?
    """

    get_feature = [lambda d: d["tokens_before"],
                   lambda d: d["tokens_after"],
                   lambda d: d["summary_tokens"],
                   lambda d: d["tokens_forgotten"],
                   lambda d: d["forgotten_n"],
                   lambda d: d["compression_ratio_forgotten"],
                   lambda d: d["vllm_delta"].map(lambda r: r["vllm:prompt_tokens_total"]),
                   lambda d: d["vllm_delta"].map(lambda r: r["vllm:generation_tokens_total"]),
                   lambda d: d["tokens_after"] / d["tokens_before"],
                   lambda d: d["tokens_before"] - d["tokens_after"]]
    
    print(df["tokens_before"].max(), df["tokens_before"].min())
    
    for f in get_feature:
        res = stats.spearmanr(df["latency_s"], f(df))
        print(f"rho = {res.statistic:.3f}, p = {res.pvalue:.3g}")


def plot_tput(runs: list[dict]):
    modes = set(d["mode"] for d in runs)
    lens = sorted(set(d["prefix_len"] for d in runs))
    
    cached_toks = []
    tok_tputs = []
    tok_tputs_errs = []
    
    for len in lens:
        trials = [d for d in runs
                  if d["prefix_len"] == len and d["mode"] == "fixed:2048"]
        
        cached_tokens_mean = np.mean([t["cached_tokens"] for t in trials])
        cached_tokens_std = np.mean([t["cached_tokens"] for t in trials])
        
        tok_tput_mean = np.mean([t["full_agg_tok_s"] for t in trials])
        tok_tput_std = np.std([t["full_agg_tok_s"] for t in trials])
        
        cached_toks.append(cached_tokens_mean)
        tok_tputs.append(tok_tput_mean)
        tok_tputs_errs.append(tok_tput_std)
        
    x_ticks = []
    x_labels = []
    for i in range(9):
        val = 1000 * 2**i
        x_ticks.append(val)
        x_labels.append(f"{val // 1000}K" if val > 5000 else "")
    
        
    plt.errorbar(cached_toks, tok_tputs, yerr=tok_tputs_errs,
                 fmt='o-', capsize=4)
    plt.grid(True)
    plt.xlabel("# cached prompt tokens")
    plt.ylabel("agg token throughput (toks/s)")
    plt.xticks(x_ticks, x_labels)
    plt.tight_layout()
    plt.show()
        


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--data", "-d", required=True)
    ap.add_argument("--plot-type", "-t", required=True,
                    choices=["lat_energy_vs_out_tok_budget",
                             "compact_cdf",
                             "compact_stats", "tput"])
    
    args = ap.parse_args()
    
    if args.plot_type == "lat_energy_vs_out_tok_budget":
        res = json.load(open(args.data))
        plot_prefill_lat_vs_out_toks(res)
        
    elif args.plot_type == "compact_cdf":
        df = pd.read_json(args.data, lines=True)
        plot_compactions_cdf(df)
        
    elif args.plot_type == "compact_stats":
        df = pd.read_json(args.data, lines=True)
        stats_compaction(df)
        
    elif args.plot_type == "tput":
        s = open(args.data).read().split("\n")
        plot_tput([json.loads(r) for r in s if len(r) > 0])

    # prefill_bench_res_path = sys.argv[1]
    
     
#     dfs = []
#     for elt in os.listdir(prefill_bench_res_path):
#        dfs.append(pd.read_csv(os.path.join(prefill_bench_res_path, elt)))
#    
#     plot_vs_cached_ratio(dfs)
    
    
    # plot_energy_usage_vs_prompt_len(pd.read_csv(prefill_bench_res_path))
    #plot_compute_transfer_lat_vs_prompt_len(pd.read_csv(prefill_bench_res_path))
