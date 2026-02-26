import { createFileRoute, Outlet } from '@tanstack/react-router'
import {
    SidebarInset,
    SidebarProvider,
    SidebarTrigger,
} from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/dashboard/app-sidebar"
import { Separator } from "@/components/ui/separator"
import { UserNav } from "@/components/dashboard/user-nav"

export const Route = createFileRoute('/dashboard')({
  component: DashboardLayout,
})

function DashboardLayout() {
  return (
    <SidebarProvider>
        <AppSidebar />
        <SidebarInset>
            <div className="flex-col md:flex h-full">
                <header className="flex md:border-b-0! h-16 shrink-0 items-center gap-2 border-b border-sidebar-border px-4">
                    <SidebarTrigger className="-ml-1 mr-2 md:hidden" />
                    <Separator
                        orientation="vertical"
                        aria-disabled
                        className="mr-4 h-4 md:hidden"
                    />
                    <div className="ml-auto flex items-center space-x-4 md:hidden">
                        <UserNav />
                    </div>
                </header>
                <div className="flex-1 space-y-4 p-8 pt-6 h-full">
                    <Outlet />
                </div>
            </div>
        </SidebarInset>
    </SidebarProvider>
  )
}
