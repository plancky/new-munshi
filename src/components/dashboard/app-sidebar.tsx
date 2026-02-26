import * as React from "react"
import { Link } from "@tanstack/react-router"
import {
  BarChart,
  Bell,
  FileText,
  Home,
  Package,
  Settings,
  Users,
} from "lucide-react"

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar"

const data = {
  navMain: [
    {
      title: "Overview",
      url: "/dashboard",
      icon: Home,
    },
    {
      title: "Customers",
      url: "/dashboard",
      icon: Users,
    },
    {
      title: "Products",
      url: "/dashboard",
      icon: Package,
    },
    {
      title: "Analytics",
      url: "/dashboard",
      icon: BarChart,
    },
    {
      title: "Reports",
      url: "/dashboard",
      icon: FileText,
    },
    {
      title: "Notifications",
      url: "/dashboard",
      icon: Bell,
    },
    {
      title: "Settings",
      url: "/dashboard",
      icon: Settings,
    },
  ],
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  return (
    <Sidebar {...props}>
      <SidebarHeader className="h-16 border-b border-sidebar-border px-6 justify-center">
        <div className="flex items-center gap-2 font-semibold">
          <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
            <Package className="size-4" />
          </div>
          <span className="truncate">SaaS Munshi</span>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Application</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {data.navMain.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild isActive={item.title === "Overview"}>
                    <Link to={item.url}>
                      <item.icon />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarRail />
    </Sidebar>
  )
}
