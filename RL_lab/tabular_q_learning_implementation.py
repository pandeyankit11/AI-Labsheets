"""Task 1-5: reproducible tabular Q-learning and evaluation."""
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from rl_core import train_q_learning, evaluate_q_policy, evaluate_random_policy
ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/"results"; RESULTS.mkdir(exist_ok=True)
def main():
    Q,training=train_q_learning(seed=7,episodes=1500)
    greedy=evaluate_q_policy(Q); additional=evaluate_q_policy(Q,start=(0,0))
    random_baseline=evaluate_random_policy(episodes=100,seed=17); window=50
    returns=np.asarray(training["returns"],float); successes=np.asarray(training["successes"],float)
    mr=np.convolve(returns,np.ones(window)/window,mode="valid")
    ms=np.convolve(successes,np.ones(window)/window,mode="valid")
    fig,ax=plt.subplots(); ax.plot(np.arange(window,len(returns)+1),mr)
    ax.set(xlabel="Episode",ylabel=f"Mean return ({window}-episode moving average)",title="Tabular Q-learning training return")
    fig.tight_layout();fig.savefig(RESULTS/"q_learning_return.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(); ax.plot(np.arange(window,len(successes)+1),ms)
    ax.set(xlabel="Episode",ylabel=f"Success rate ({window}-episode moving average)",ylim=(0,1.05),title="Tabular Q-learning training success")
    fig.tight_layout();fig.savefig(RESULTS/"q_learning_success.png",dpi=160);plt.close(fig)
    with (RESULTS/"q_learning_episode_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["episode","return","success","timeout","invalid_moves","epsilon"])
        w.writerows(zip(range(1,len(returns)+1),training["returns"],training["successes"],training["timeouts"],training["invalid_moves"],training["epsilons"]))
    summary={"training":{"seed":7,"episodes":1500,"alpha":0.2,"gamma":0.95,"epsilon_start":training["epsilons"][0],
       "epsilon_after_decay":training["epsilons"][1199],"moving_average_window":window,
       "final_50_episode_success_rate":float(np.mean(training["successes"][-50:])),
       "final_50_episode_mean_return":float(np.mean(training["returns"][-50:]))},
       "greedy_default_start":greedy,"greedy_additional_start_0_0":additional,
       "random_policy_100_episodes":random_baseline,"shortest_route_steps":4,
       "shortest_route":[[2,0],[2,1],[2,2],[2,3],[2,4]]}
    (RESULTS/"q_learning_results.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    np.savetxt(RESULTS/"q_table.csv",Q,delimiter=",")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
