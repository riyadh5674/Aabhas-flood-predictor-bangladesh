import { cn } from "@/lib/cn";

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
  className,
}: {
  eyebrow?: string;
  title: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <header className={cn("mb-7 flex flex-wrap items-end justify-between gap-4", className)}>
      <div className="min-w-0">
        {eyebrow ? <div className="micro mb-2.5">{eyebrow}</div> : null}
        <h1 className="display text-[clamp(1.65rem,2.6vw,2.25rem)]">{title}</h1>
        {description ? (
          <p className="mt-2 max-w-[62ch] text-[14px] leading-relaxed text-2">{description}</p>
        ) : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </header>
  );
}

export function SectionTitle({
  children,
  hint,
  action,
}: {
  children: React.ReactNode;
  hint?: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <div className="mb-4 flex items-end justify-between gap-4">
      <div>
        <h2 className="display text-[17px] tracking-[-0.02em]">{children}</h2>
        {hint ? <p className="mt-1 text-[12.5px] muted">{hint}</p> : null}
      </div>
      {action}
    </div>
  );
}
