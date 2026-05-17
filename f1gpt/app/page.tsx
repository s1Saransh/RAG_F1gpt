"use client";

import { FormEvent, useState } from "react";
import { ChatBubble } from "@/components/ChatBubble";
import { ChatComposer } from "@/components/ChatComposer";
import { PromptSuggestions } from "@/components/PromptSuggestions";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

const promptSuggestions = [
  "Summarize the history of Formula 1.",
  "Compare Lewis Hamilton and Max Verstappen.",
  "How does F1 qualifying work?",
  "What makes Ferrari important in F1?",
];

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function sendQuestion(questionText: string) {
    const question = questionText.trim();
    if (!question || isLoading) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
    };
    const assistantMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
    };
    const nextMessages = [...messages, userMessage];

    setMessages([...nextMessages, assistantMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: nextMessages }),
      });

      if (!response.ok || !response.body) {
        throw new Error("Chat request failed");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        const content = lines
          .filter((line) => line.startsWith("0:"))
          .map((line) => JSON.parse(line.slice(2)) as string)
          .join("");

        if (content) {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantMessage.id
                ? { ...message, content: message.content + content }
                : message
            )
          );
        }
      }
    } catch {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantMessage.id
            ? {
                ...message,
                content: "Sorry, I could not get an answer right now.",
              }
            : message
        )
      );
    } finally {
      setIsLoading(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendQuestion(input);
  }

  return (
    <main className="min-h-screen bg-slate-100 px-4 py-8 text-slate-950">
      <section className="mx-auto flex min-h-[calc(100vh-4rem)] w-full max-w-3xl flex-col rounded-lg border border-slate-200 bg-white shadow-sm">
        <header className="border-b border-slate-200 px-5 py-4">
          <h1 className="text-xl font-semibold">F1GPT Chat</h1>
          <p className="mt-1 text-sm text-slate-500">
            Ask questions against the Formula 1 data you seeded into MongoDB.
          </p>
        </header>

        <div className="flex-1 overflow-y-auto px-5 py-6">
          {messages.length === 0 ? (
            <div className="flex h-full min-h-80 items-center justify-center">
              <div className="w-full max-w-xl text-center">
                <h2 className="text-2xl font-semibold">
                  What would you like to know?
                </h2>
                <p className="mt-2 text-sm text-slate-500">
                  Start with a prompt suggestion or type your own question.
                </p>
                <PromptSuggestions
                  disabled={isLoading}
                  onSelect={(prompt) => void sendQuestion(prompt)}
                  prompts={promptSuggestions}
                />
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((msg) => (
                <ChatBubble
                  key={msg.id}
                  content={msg.content}
                  isLoading={isLoading && msg.role === "assistant" && !msg.content}
                  role={msg.role}
                />
              ))}
            </div>
          )}
        </div>

        <ChatComposer
          input={input}
          isLoading={isLoading}
          onInputChange={setInput}
          onSubmit={handleSubmit}
        />
      </section>
    </main>
  );
}
