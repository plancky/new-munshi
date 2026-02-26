import Logo from "@/logo/logo.svg?react";
import { Link, useLocation } from "@tanstack/react-router";
import { BarChart, Home, Package, Settings, Users } from "lucide-react";
import * as React from "react";

import { NavUser } from "@/components/dashboard/nav-user";
import {
    Sidebar,
    SidebarContent,
    SidebarFooter,
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarHeader,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
} from "@/components/ui/sidebar";
import { BooksIcon, VinylRecordIcon } from "@phosphor-icons/react/dist/ssr";

const data = {
    user: {
        name: "chirag",
        email: "chirag@trymunshi.com",
        avatar: "/avatars/01.png",
    },
    navMain: [
        {
            title: "Overview",
            url: "/dashboard",
            icon: Home,
        },
        {
            title: "Whispers",
            url: "/dashboard/whispers",
            icon: Users,
        },
        {
            title: "Tangents",
            url: "/dashboard/tangents",
            icon: Package,
        },
        {
            title: "Insights",
            url: "/dashboard/insights",
            icon: BarChart,
        },
    ],
    dataRoomNav: [
        {
            title: "Transcripts",
            url: "/dashboard/transcripts",
            icon: BooksIcon,
        },
        {
            title: "Cassetes",
            url: "/dashboard/cassetes",
            icon: VinylRecordIcon,
        },
    ],
    navSettings: [
        {
            title: "Settings",
            url: "/dashboard/settings",
            icon: Settings,
        },
    ],
};

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
    const location = useLocation();
    const pathname = location.pathname;

    return (
        <Sidebar {...props}>
            <SidebarHeader className="h-16 px-6 justify-center">
                <div className="flex items-center justify-start gap-2 font-semibold">
                    <Logo className="h-8 text-foreground" />
                    <h1 className="text-xl">Munshi</h1>
                </div>
            </SidebarHeader>
            <SidebarContent>
                <SidebarGroup>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            {data.navMain.map((item) => (
                                <SidebarMenuItem key={item.title}>
                                    <SidebarMenuButton
                                        asChild
                                        isActive={
                                            pathname === item.url ||
                                            (item.url === "/dashboard" &&
                                                pathname === "/dashboard/")
                                        }
                                    >
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
                <SidebarGroup>
                    <SidebarGroupLabel>Data Room</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            {data.dataRoomNav.map((item) => (
                                <SidebarMenuItem key={item.title}>
                                    <SidebarMenuButton
                                        asChild
                                        isActive={pathname === item.url}
                                    >
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
            <SidebarFooter>
                <SidebarGroup>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            {data.navSettings.map((item) => (
                                <SidebarMenuItem key={item.title}>
                                    <SidebarMenuButton
                                        asChild
                                        isActive={pathname === item.url}
                                    >
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
                <NavUser user={data.user} />
            </SidebarFooter>
        </Sidebar>
    );
}
