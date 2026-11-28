// Mirrors tuncall.record.CallRecord and tuncall.serve.db.Call (D-55a, D-55c).

export type Status = "queued" | "processing" | "done" | "failed";

export interface Entity {
  type: string;
  value: string;
}

export interface Segment {
  start: number;
  end: number;
  speaker: string;
  role: "agent" | "customer" | "unknown";
  text_raw: string;
  text_normalized: string;
  asr_model_version: string;
  intent: string | null;
  entities: Entity[];
  sentiment: "positive" | "negative" | null;
}

export interface CallRecord {
  call_id: string;
  source: string;
  duration_s: number;
  sample_rate: number;
  channels: number;
  segments: Segment[];
  meta: {
    pipeline_version: string;
    diarization_model_version: string;
    processing_ms_by_stage: Record<string, number>;
    nlp_model_versions: Record<string, string>;
  };
}

export interface Call {
  id: string;
  filename: string;
  status: Status;
  attempts: number;
  num_speakers: number | null;
  error: string | null;
  created_at: string;
  claimed_at: string | null;
  finished_at: string | null;
  record: CallRecord | null;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail;
    throw new Error(typeof detail === "string" ? detail : `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const listCalls = () => fetch("/calls").then((r) => json<Call[]>(r));

export const getCall = (id: string) => fetch(`/calls/${id}`).then((r) => json<Call>(r));

export const audioUrl = (id: string) => `/calls/${id}/audio`;

export async function uploadCall(file: File, numSpeakers: number): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  form.append("num_speakers", String(numSpeakers));
  const res = await fetch("/calls", { method: "POST", body: form });
  return (await json<{ call_id: string }>(res)).call_id;
}

export async function deleteCall(id: string): Promise<void> {
  const res = await fetch(`/calls/${id}`, { method: "DELETE" });
  if (!res.ok && res.status !== 404) throw new Error(`HTTP ${res.status}`);
}
