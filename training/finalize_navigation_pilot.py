"""Summarize completed course comparisons after a user-stopped PPO evaluation."""
import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO

from training.train_navigation import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    output = args.run.resolve()
    read = lambda name: json.loads((output / name).read_text(encoding="utf-8"))
    config, before, after = read("config.json"), read("pre-evaluation.json"), read("post-evaluation.json")
    # Include only courses whose pre AND post evaluations have finished.
    completed = {r["course"] for r in after}
    before = [r for r in before if r["course"] in completed]
    updates, episodes = read("update-metrics.json"), read("train-episodes.json")
    if not before or not updates:
        raise ValueError("No completed comparison or PPO updates")
    torch.set_num_threads(2)
    initial = PPO.load(output / "initial-policy", device="cpu")
    final = PPO.load(output / "final-policy", device="cpu")
    old = dict(initial.policy.named_parameters())
    changed = sum(not torch.equal(old[k], p) for k, p in final.policy.named_parameters())
    finite = all(torch.isfinite(p).all() for p in final.policy.parameters())
    if not changed or not finite or digest(config["perception"]) != config["perception_sha256"]:
        raise AssertionError("Policy update/frozen perception check failed")
    metrics = dict(actual_steps=int(updates[-1]["step"]),
                   policy_parameters=sum(p.numel() for p in final.policy.parameters()),
                   changed_parameter_tensors=changed, updates=updates, training_episodes=episodes,
                   pre=before, post=after, perception_unchanged=True,
                   # Original runner verifies save/load before starting post evaluation.
                   save_load_match=True, evaluation_stopped_by_user=True,
                   scope="Only fully completed pre/post course pairs; unfinished comparison excluded")
    write_json(output / "metrics.json", metrics)
    rows = []
    for b, a in zip(before, after):
        for r, label in ((b, "학습 전"), (a, "학습 후")):
            rows.append(f"| {label} | {r['waypoints_reached']}/2 | {r['route_progress_m']:.2f}m | "
                        f"{r['episode_return']:.2f} | {r['min_clearance_m']:.2f}m |")
    text = "\n".join([
        "# 첫 PPO 실험: 완료된 코스만 비교", "",
        f"탑뷰 인식을 고정하고 주행 정책을 실제 행동 {metrics['actual_steps']:,}회, PPO {len(updates)}묶음으로 학습했다.",
        "자기 상태·경유지 좌표는 직접 전달한다. 기본 에이전트 시범이나 수동 조작은 사용하지 않았다.", "",
        "사용자 요청으로 추가 검증 주행을 중단했다. 완전히 기록된 첫 코스의 같은 시드·80초 주행만 비교한다.", "",
        "| 구분 | 경유지 | 남은 경로 거리 감소 | 누적 보상 | 최소 여유 거리 |",
        "|---|---|---:|---:|---:|", *rows, "",
        "두 주행 모두 80초 실험 중단으로 미완주다. 충돌 없이 끝났지만 첫 경유지를 놓쳤다. 공식 제한 600초 평가 결과가 아니다.",
        "**이 짧은 실험에서는 주행 성능 개선을 확인하지 못했다.** 학습 연결·가중치 갱신·저장/재로딩은 정상이다.",
        "2,048회 수집에서 끝까지 진행된 학습 에피소드는 2개뿐이어서, 학습량만으로 장기 경유지 추종을 익혔다고 판단할 수 없다.",
        "다음 수정 후보는 공개 경유지 좌표로 계산한 추종 행동에 PPO 보정량을 더하는 구조다. 먼저 경유지 추종을 안정화한 뒤 회피를 학습한다. 이는 아직 실행하지 않은 제안이다.", "",
        "[자세한 요약과 궤적](summary-ko.md) · [수치](metrics.json) · [체크포인트](final-policy.zip)", ""])
    (output / "report.md").write_text(text, encoding="utf-8")
    print(output / "report.md")


if __name__ == "__main__":
    main()
