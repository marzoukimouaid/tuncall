import type { Status } from "./api";

export const POLL_MS = 3000; // D-50b: the dashboard polls; no WebSockets

const STATUS_CLASS: Record<Status, string> = {
  queued: "bg-slate-200 text-slate-800",
  processing: "bg-amber-100 text-amber-900",
  done: "bg-green-100 text-green-900",
  failed: "bg-red-100 text-red-900",
};

export function StatusBadge({ status }: { status: Status }) {
  return (
    <span className={`rounded px-2 py-0.5 text-xs font-medium ${STATUS_CLASS[status]}`}>
      {status}
    </span>
  );
}
