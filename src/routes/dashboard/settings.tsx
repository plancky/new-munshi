import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/dashboard/settings')({
  component: SettingsPage,
})

function SettingsPage() {
  return (
    <div className="flex items-center justify-center h-full">
      <h1 className="text-2xl font-bold">Settings</h1>
    </div>
  )
}
