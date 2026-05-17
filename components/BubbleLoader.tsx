type BubbleLoaderProps = {
  tone?: "dark" | "light";
};

export function BubbleLoader({ tone = "dark" }: BubbleLoaderProps) {
  const dotColor = tone === "light" ? "bg-white" : "bg-slate-500";

  return (
    <span className="flex items-center gap-1" aria-label="Loading">
      <span
        className={`h-1.5 w-1.5 animate-bounce rounded-full ${dotColor}`}
        style={{ animationDelay: "-0.2s" }}
      />
      <span
        className={`h-1.5 w-1.5 animate-bounce rounded-full ${dotColor}`}
        style={{ animationDelay: "-0.1s" }}
      />
      <span className={`h-1.5 w-1.5 animate-bounce rounded-full ${dotColor}`} />
    </span>
  );
}
