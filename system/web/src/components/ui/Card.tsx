import { cn } from "@/lib/cn";

export function Card({
  children,
  className,
  hover = false,
  flat = false,
  ...rest
}: {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  flat?: boolean;
} & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("card", hover && "card--hover", flat && "card--flat", className)}
      {...rest}
    >
      {children}
    </div>
  );
}

export function Skeleton({ className, style }: { className?: string; style?: React.CSSProperties }) {
  return <div className={cn("skel", className)} style={style} aria-hidden="true" />;
}

export function SkeletonText({ lines = 3, className }: { lines?: number; className?: string }) {
  return (
    <div className={cn("space-y-2", className)} aria-hidden="true">
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} className="h-3" style={{ width: i === lines - 1 ? "62%" : "100%" }} />
      ))}
    </div>
  );
}
