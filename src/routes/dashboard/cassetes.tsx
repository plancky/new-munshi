import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/dashboard/cassetes')({
  component: CassetesPage,
})

function CassetesPage() {
  return (
    <div className="flex items-center justify-center h-full">
      <h1 className="text-2xl font-bold">Cassetes</h1>
    </div>
  )
}
