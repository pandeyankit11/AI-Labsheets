"""Task 6: single-robot DQN."""
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from rl_core import train_single_dqn
ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/"results";RESULTS.mkdir(exist_ok=True)
def main():
    network,raw=train_single_dqn(seed=7,total_steps=12000)
    suc,tmo,inv,ret=raw["episode_successes"],raw["episode_timeouts"],raw["episode_invalid_moves"],raw["episode_returns"]
    n=len(ret);w=50;ms=np.convolve(np.asarray(suc,float),np.ones(w)/w,mode="valid");mr=np.convolve(np.asarray(ret,float),np.ones(w)/w,mode="valid")
    fig,ax=plt.subplots();ax.plot(np.arange(w,n+1),ms);ax.set(xlabel="Completed episode",ylabel="50-episode success moving average",ylim=(0,1.05));fig.tight_layout();fig.savefig(RESULTS/"single_dqn_success.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots();ax.plot(np.arange(w,n+1),mr);ax.set(xlabel="Completed episode",ylabel="50-episode mean return");fig.tight_layout();fig.savefig(RESULTS/"single_dqn_return.png",dpi=160);plt.close(fig)
    with (RESULTS/"single_dqn_episode_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        cw=csv.writer(f);cw.writerow(["episode","return","success","timeout","invalid_moves"]);cw.writerows(zip(range(1,n+1),ret,suc,tmo,inv))
    summary={"config":raw["config"],"completed_episodes":n,"training_success_rate":float(np.mean(suc)),
      "training_timeout_rate":float(np.mean(tmo)),"training_mean_invalid_moves":float(np.mean(inv)),
      "final_50_episode_success_rate":float(np.mean(suc[-50:])),"optimizer_updates":raw["optimizer_updates"],
      "target_network_syncs":raw["target_sync_updates_requested"],"last_loss":raw["loss_last"],"evaluation":raw["evaluation"]}
    (RESULTS/"single_robot_dqn_results.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
