"use client";

export default function ErrorPage({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <main className="fatal-shell">
      <section className="fatal-card">
        <p className="eyebrow">ARCHTRACE</p>
        <h1>The explorer hit an unexpected error.</h1>
        <p>{error.message || "Unknown application error."}</p>
        <button className="primary-button" type="button" onClick={reset}>
          Reload explorer
        </button>
      </section>
    </main>
  );
}
