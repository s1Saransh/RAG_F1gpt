import { FormEvent } from "react";
import { BubbleLoader } from "./BubbleLoader";

type ChatComposerProps = {
  input: string;
  isLoading: boolean;
  onInputChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
};

export function ChatComposer({
  input,
  isLoading,
  onInputChange,
  onSubmit,
}: ChatComposerProps) {
  return (
    <form onSubmit={onSubmit} className="flex gap-3 border-t border-slate-200 p-4">
      <input
        id="chat-input"
        type="text"
        value={input}
        onChange={(event) => onInputChange(event.target.value)}
        placeholder="Type your question here..."
        className="min-w-0 flex-1 rounded-lg border border-slate-300 bg-white px-4 py-3 text-sm outline-none transition focus:border-slate-500 focus:ring-2 focus:ring-slate-200"
        disabled={isLoading}
      />
      <button
        id="chat-submit"
        type="submit"
        disabled={isLoading || !input.trim()}
        className="flex min-w-20 items-center justify-center rounded-lg bg-slate-900 px-5 py-3 text-sm font-medium text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {isLoading ? <BubbleLoader tone="light" /> : "Send"}
      </button>
    </form>
  );
}
