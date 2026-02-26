import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/dashboard/insights')({
  component: InsightsPage,
})

function InsightsPage() {
  return (
    <div className="flex items-center justify-center h-full">
      <h1 className="text-2xl font-bold">Insights</h1>
    </div>
  )
}
