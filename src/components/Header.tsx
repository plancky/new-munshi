import Logo from "@/logo/logo.svg?react";
import { BooksIcon, StackIcon } from "@phosphor-icons/react/dist/ssr";

export default function Header() {
    return (
        <div className="relative z-50">
            <header className="content-grid fixed w-full py-4">
                <div className="content lg:breakout flex items-center justify-between rounded-xl border border-border/60 bg-background/70 px-4 py-3 shadow backdrop-blur-sm transition-colors">
                    <a href="/" className="group flex items-center gap-3">
                        <span className="font-normal transition-transform duration-200 group-hover:scale-105">
                            <Logo className="h-8 text-foreground" />
                        </span>
                        <span className="hidden font-heading text-base font-medium text-foreground lg:block">
                            Munshi
                        </span>
                    </a>

                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2">
                            <a
                                href="/catalogue"
                                className="flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted/50 hover:text-foreground"
                            >
                                <BooksIcon className="h-4 w-4" />
                                <span className="hidden sm:block">
                                    Catalogue
                                </span>
                            </a>
                            <a
                                href="/collections"
                                className="flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted/50 hover:text-foreground"
                            >
                                <StackIcon className="h-4 w-4" />
                                <span className="hidden sm:block">
                                    Collections
                                </span>
                            </a>
                            {/* <a
                                href="/ask"
                                className="flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted/50 hover:text-foreground"
                            >
                                <StackIcon className="h-4 w-4" />
                                <span className="hidden sm:block">
                                    Ask Munshi
                                </span>
                            </a> */}
                        </div>
                        <div className="h-4 w-px bg-border/60" />
                    </div>
                </div>
            </header>
        </div>
    );
}
