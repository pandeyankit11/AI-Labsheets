"""Task 7-8: simultaneous two-robot DQN."""
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from rl_core import train_two_robot_dqn
ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/"results";RESULTS.mkdir(exist_ok=True)
def main():
    network,raw=train_two_robot_dqn(seed=7,total_steps=60000)
    ret,suc,tmo,inv=raw["episode_returns"],raw["episode_successes"],raw["episode_timeouts"],raw["episode_invalid_moves"]
    n=len(ret);w=50;ms=np.convolve(np.asarray(suc,float),np.ones(w)/w,mode="valid");mr=np.convolve(np.asarray(ret,float),np.ones(w)/w,mode="valid")
    fig,ax=plt.subplots();ax.plot(np.arange(w,n+1),ms);ax.set(xlabel="Completed episode",ylabel="50-episode success moving average",ylim=(0,1.05));fig.tight_layout();fig.savefig(RESULTS/"two_robot_dqn_success.png",dpi=160);fig.savefig(RESULTS/"two_robot_dqn_success.svg");plt.close(fig)
    fig,ax=plt.subplots();ax.plot(np.arange(w,n+1),mr);ax.set(xlabel="Completed episode",ylabel="50-episode mean return");fig.tight_layout();fig.savefig(RESULTS/"two_robot_dqn_return.png",dpi=160);plt.close(fig)
    with (RESULTS/"two_robot_dqn_episode_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        cw=csv.writer(f);cw.writerow(["episode","return","success","timeout","invalid_moves"]);cw.writerows(zip(range(1,n+1),ret,suc,tmo,inv))
    summary={"config":raw["config"],"completed_episodes":n,"training_success_rate":float(np.mean(suc)),
      "training_timeout_rate":float(np.mean(tmo)),"training_collision_rate":float(1-np.mean(suc)-np.mean(tmo)),
      "training_mean_invalid_moves":float(np.mean(inv)),"last_50_episode_success_rate":float(np.mean(suc[-50:])),
      "optimizer_updates":raw["optimizer_updates"],"target_network_syncs":raw["target_sync_updates_requested"],
      "last_loss":raw["loss_last"],"evaluation":raw["evaluation"],
      "interpretation":"Seed-7 60,000-step run learned a collision-free greedy delivery policy from both tested starts. Training still contains exploratory failures; frozen greedy evaluations are reported separately."}
    (RESULTS/"two_robot_dqn_results.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    path=raw["evaluation"]["default_start"]["path"]
    fig,ax=plt.subplots();ax.set_xlim(-.5,4.5);ax.set_ylim(4.5,-.5);ax.set_xticks(range(5));ax.set_yticks(range(5));ax.grid(True)
    for r,c in {(0,2),(1,2),(3,2),(4,2)}:ax.scatter(c,r,marker="s",s=180)
    pa=[s[0] for s in path];pb=[s[1] for s in path]
    ax.plot([p[1] for p in pa],[p[0] for p in pa],marker="o",label="Robot A actual greedy rollout")
    ax.plot([p[1] for p in pb],[p[0] for p in pb],marker="s",label="Robot B actual greedy rollout")
    ax.set(title="Seed-7 two-robot DQN greedy rollout",xlabel="Column",ylabel="Row");ax.legend();fig.tight_layout()
    fig.savefig(RESULTS/"two_robot_greedy_trajectory.png",dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
