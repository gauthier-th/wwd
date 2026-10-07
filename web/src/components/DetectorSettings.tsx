import { useState } from "react";

import type { OperatingPoint, Report } from "../api";
import { falseAlarmRate } from "../format";
import type { ModelKind } from "./LiveTest";

export interface DetectorChoice {
  point: OperatingPoint;
  kind: ModelKind;
  gate: boolean;
}

/** Sensitivity (one of the training's calibrated operating points), model file and speech gate. */
export function useDetectorChoice(report: Report) {
  const points = report.operating_points;
  const [index, setIndex] = useState(() =>
    Math.max(0, points.findIndex((p) => p.fa_per_hour === report.target_fa_per_hour)));
  const [kind, setKind] = useState<ModelKind>("fp32");
  const [gate, setGate] = useState(true);
  return { choice: { point: points[index], kind, gate }, points, index, setIndex, setKind, setGate };
}

export function DetectorSettings({ state, disabled }: { state: ReturnType<typeof useDetectorChoice>; disabled?: boolean }) {
  const { choice, points, index, setIndex, setKind, setGate } = state;
  const { point, kind, gate } = choice;
  return (
    <>
      <div className="field">
        <label htmlFor="sensitivity">Sensitivity</label>
        <input id="sensitivity" type="range" min={0} max={points.length - 1} step={1} value={index}
               disabled={disabled} onChange={(e) => setIndex(Number(e.target.value))} data-testid="sensitivity"
               aria-valuetext={`about ${falseAlarmRate(point.fa_per_hour)} false alarms`} />
        <div className="row small muted">
          <span>Fewer false alarms</span>
          <span className="spacer" />
          <span>More sensitive</span>
        </div>
        <div className="small">
          About <strong>{falseAlarmRate(point.fa_per_hour)}</strong> false alarm
          {point.fa_per_hour > 1 ? "s" : ""} in conversation (threshold {point[kind].threshold.toFixed(3)}).
        </div>
      </div>
      <div className="row" style={{ margin: "12px 0" }}>
        <div className="segmented" role="group" aria-label="Model file">
          <button type="button" aria-pressed={kind === "fp32"} disabled={disabled} onClick={() => setKind("fp32")}>Full</button>
          <button type="button" aria-pressed={kind === "int8"} disabled={disabled} onClick={() => setKind("int8")}>
            Small (int8)
          </button>
        </div>
        <label className="check small">
          <input type="checkbox" checked={gate} disabled={disabled} onChange={(e) => setGate(e.target.checked)}
                 data-testid="gate" />
          Only listen while someone speaks
        </label>
      </div>
    </>
  );
}
