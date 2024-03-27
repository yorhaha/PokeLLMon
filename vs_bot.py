import asyncio
import time
from tqdm import tqdm
import numpy as np
from poke_env import AccountConfiguration, ShowdownServerConfiguration
import os
import pickle as pkl
import argparse
from dotenv import load_dotenv

from poke_env.player import LLMPlayer, SimpleHeuristicsPlayer

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="gpt-4-0125-preview",
    choices=["gpt-3.5-turbo-0125", "gpt-4-1106-preview", "gpt-4-0125-preview", "Llama-2-7b-chat-hf"],
)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--prompt_algo", default="sc", choices=["io", "sc", "cot", "tot"])
# parser.add_argument("--log_dir", type=str, default="./battle_log/pokellmon_vs_bot")
parser.add_argument("--api_key", type=str, default=os.environ.get("API_KEY"))
parser.add_argument("--api_base", type=str, default=os.environ.get("API_BASE"))
parser.add_argument("--username", type=str, default=os.environ.get("USERNAME"))
parser.add_argument("--password", type=str, default=os.environ.get("PASSWORD"))
parser.add_argument("--n_battles", type=int, default=20)
args = parser.parse_args()

log_dir = f"./battle_log/pokellmon_vs_bot/{args.model}_{args.temperature}_{args.prompt_algo}"


async def main():

    heuristic_player = SimpleHeuristicsPlayer(battle_format="gen8randombattle")

    os.makedirs(log_dir, exist_ok=True)
    llm_player = LLMPlayer(
        battle_format="gen8randombattle",
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        temperature=args.temperature,
        prompt_algo=args.prompt_algo,
        log_dir=log_dir,
        account_configuration=AccountConfiguration(args.username, args.password),
        save_replays=log_dir,
    )

    # dynamax is disabled for local battles.
    heuristic_player._dynamax_disable = True
    llm_player._dynamax_disable = True

    # play against bot for five battles
    for i in tqdm(range(args.n_battles)):
        await llm_player.battle_against(heuristic_player, n_battles=1)
        # x = np.random.randint(0, 100)
        # if x > 50:
        #     await heuristic_player.battle_against(llm_player, n_battles=1)
        # else:
        #     await llm_player.battle_against(heuristic_player, n_battles=1)
        for battle_id, battle in llm_player.battles.items():
            with open(f"{log_dir}/{battle_id}.pkl", "wb") as f:
                pkl.dump(battle, f)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
