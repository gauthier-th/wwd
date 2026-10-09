import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { api, type ProgressEvent, type Training } from "./api";
import { subscribe } from "./local/backend";

export const FINAL_STATUSES = ["done", "failed", "cancelled"] as const;
export const isFinal = (t: Training) => (FINAL_STATUSES as readonly string[]).includes(t.status);

export const useLanguages = () =>
  useQuery({ queryKey: ["languages"], queryFn: api.languages, staleTime: Infinity });

export const useBaseModels = () =>
  useQuery({ queryKey: ["base-models"], queryFn: api.baseModels, staleTime: 60_000 });

export const useWakeWord = (id: string) =>
  useQuery({ queryKey: ["wakeword", id], queryFn: () => api.wakeWord(id) });

export const useTraining = (id: string) =>
  useQuery({
    queryKey: ["training", id],
    queryFn: () => api.training(id),
    // A training another tab is running sends no events here: check on it now and then.
    refetchInterval: (q) => (q.state.data && !isFinal(q.state.data) ? 5000 : false),
  });

/** Progress of a training running in this tab; refreshes the training (and its wake word) at the end. */
export function useTrainingEvents(training: Training | undefined) {
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const client = useQueryClient();
  const active = training !== undefined && !isFinal(training);
  const id = training?.id;
  const wakeWordId = training?.wakeword_id;

  useEffect(() => {
    if (!active || !id) return;
    setEvents([]); // the subscription replays every event from the start
    return subscribe(
      id,
      (event) => setEvents((prev) => [...prev, event]),
      () => {
        client.invalidateQueries({ queryKey: ["training", id] });
        client.invalidateQueries({ queryKey: ["wakeword", wakeWordId] });
      },
    );
  }, [active, id, wakeWordId, client]);

  return events;
}
