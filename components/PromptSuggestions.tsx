type PromptSuggestionsProps = {
  disabled?: boolean;
  onSelect: (prompt: string) => void;
  prompts: string[];
};

export function PromptSuggestions({
  disabled = false,
  onSelect,
  prompts,
}: PromptSuggestionsProps) {
  return (
    <div className="mt-6 grid gap-2 sm:grid-cols-2">
      {prompts.map((prompt) => (
        <button
          key={prompt}
          type="button"
          onClick={() => onSelect(prompt)}
          disabled={disabled}
          className="rounded-lg border border-slate-200 bg-white px-4 py-3 text-left text-sm text-slate-700 transition hover:border-slate-400 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {prompt}
        </button>
      ))}
    </div>
  );
}
