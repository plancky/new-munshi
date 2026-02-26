import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/dashboard/transcripts')({
  component: TranscriptsPage,
})

function TranscriptsPage() {
  return (
    <div className="flex items-center justify-center h-full">
      <h1 className="text-2xl font-bold">Transcripts</h1>
    </div>
  )
}
