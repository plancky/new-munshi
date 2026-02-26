import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/dashboard/tangents')({
  component: TangentsPage,
})

function TangentsPage() {
  return (
    <div className="flex items-center justify-center h-full">
      <h1 className="text-2xl font-bold">Tangents</h1>
    </div>
  )
}
