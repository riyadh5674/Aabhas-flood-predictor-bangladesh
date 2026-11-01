import { Compass } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { useEffect } from "react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card } from "@/components/ui/Card";
import { Reveal } from "@/components/ui/Reveal";
import { StatusPill } from "@/components/StatusPill";

export function NotFoundPage() {
  const navigate = useNavigate();

  useEffect(() => {
    // no-op hook to keep structure consistent; navigate home on Enter handled globally
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center px-6">
      <Reveal>
        <div className="grid h-20 w-20 place-items-center rounded-full mb-4" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
          <Compass size={30} style={{ color: "var(--muted)" }} />
        </div>
        <PageHeader
          eyebrow="Page not found"
          title="This page doesn't exist"
          description="We can’t find that page. Head back to the Bangladesh outlook or open the risk map."
          actions={
            <>
              <Link to="/" className="btn btn--primary">
                Back to overview <span aria-hidden="true">&rarr;</span>
              </Link>
              <Link to="/map" className="btn btn--ghost">
                Open risk map
              </Link>
              <StatusPill compact />
            </>
          }
        />
      </Reveal>
    </div>
  );
}

export default NotFoundPage;
