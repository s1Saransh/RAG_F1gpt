import { BubbleLoader } from "./BubbleLoader";

type ChatBubbleProps = {
  content: string;
  isLoading?: boolean;
  role: "user" | "assistant";
};

export function ChatBubble({ content, isLoading = false, role }: ChatBubbleProps) {
  const isUser = role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-lg px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? "bg-slate-900 text-white"
            : "border border-slate-200 bg-slate-50 text-slate-800"
        }`}
      >
        <span className="mb-1 block text-xs font-medium opacity-70">
          {isUser ? "You" : "F1GPT"}
        </span>
        {isLoading ? <BubbleLoader /> : content}
      </div>
    </div>
  );
}
