from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path


def make_episode_seed(job_seed: int, episode_index: int) -> int:
    rng = random.Random((job_seed << 20) ^ (episode_index * 2654435761))
    return rng.randint(1, (1 << 63) - 1)


def winner_to_score(winner: object, shared_victory: bool, candidate_player: int) -> float:
    if shared_victory or winner is None:
        return 0.5
    try:
        wid = int(winner)
    except (TypeError, ValueError):
        return 0.5
    return 1.0 if wid == candidate_player else 0.0


def ci95(win_rate: float, n: int) -> float:
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, float(win_rate)))
    return 1.96 * math.sqrt(max(1e-12, p * (1.0 - p) / n))


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B evaluate splendor hash logic impact.")
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--simulations", type=int, default=600)
    parser.add_argument("--job-seed", type=int, default=20260412)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="games/splendor/model/best_model.onnx")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    debug_service_dir = root / "games" / "splendor" / "debug_service"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(debug_service_dir) not in sys.path:
        sys.path.insert(0, str(debug_service_dir))

    import cpp_splendor_engine_v1 as cpp_splendor_engine  # type: ignore

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (root / model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    search_options = {
        "stop_on_draw_transition": True,
        "enable_draw_chance": True,
        "chance_expand_cap": 10,
        "max_search_depth": 50,
        "max_episode_plies": 200,
    }

    records: list[dict[str, object]] = []
    total_score = 0.0
    first_score = 0.0
    second_score = 0.0
    first_n = 0
    second_n = 0

    games = max(2, int(args.games))
    for i in range(games):
        seed = make_episode_seed(int(args.job_seed) ^ 0xABCDEF, i)
        if i % 2 == 0:
            candidate_player = 0
            raw = cpp_splendor_engine.run_arena_match_fast(
                int(seed),
                "netmcts",
                int(args.simulations),
                0.0,
                str(model_path),
                "heuristic",
                int(args.simulations),
                0.0,
                None,
                dict(search_options),
                dict(search_options),
            )
        else:
            candidate_player = 1
            raw = cpp_splendor_engine.run_arena_match_fast(
                int(seed),
                "heuristic",
                int(args.simulations),
                0.0,
                None,
                "netmcts",
                int(args.simulations),
                0.0,
                str(model_path),
                dict(search_options),
                dict(search_options),
            )
        score = winner_to_score(raw.get("winner"), bool(raw.get("shared_victory", False)), candidate_player)
        total_score += score
        if candidate_player == 0:
            first_score += score
            first_n += 1
        else:
            second_score += score
            second_n += 1
        records.append(
            {
                "index": i,
                "seed": int(seed),
                "candidate_player": candidate_player,
                "winner": raw.get("winner"),
                "shared_victory": bool(raw.get("shared_victory", False)),
                "scores": raw.get("scores"),
                "candidate_score": score,
            }
        )
        if (i + 1) % 20 == 0 or (i + 1) == games:
            print(f"[ab-eval:{args.label}] progress {i + 1}/{games}", flush=True)

    win_rate = total_score / games
    first_rate = first_score / max(1, first_n)
    second_rate = second_score / max(1, second_n)
    summary = {
        "label": args.label,
        "games": games,
        "simulations": int(args.simulations),
        "job_seed": int(args.job_seed),
        "model_path": str(model_path),
        "win_rate": win_rate,
        "ci95": ci95(win_rate, games),
        "first_player_games": first_n,
        "first_player_win_rate": first_rate,
        "second_player_games": second_n,
        "second_player_win_rate": second_rate,
    }
    out = {"summary": summary, "records": records}
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (root / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
