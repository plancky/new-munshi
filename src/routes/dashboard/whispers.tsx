import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/dashboard/whispers')({
  component: WhispersPage,
})

function WhispersPage() {
  return (
    <div className="flex items-center justify-center h-full">
      <h1 className="text-2xl font-bold">Whispers</h1>
    </div>
  )
}
