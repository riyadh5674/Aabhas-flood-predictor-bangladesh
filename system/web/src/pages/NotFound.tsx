import { Compass, Home, Map } from "lucide-react";
import { Link } from "react-router-dom";
import { Card } from "@/components/ui/Card";

export function NotFoundPage() {
  return (
    <div className="flex min-h-[65vh] items-center justify-center px-4 py-10">
      <Card className="max-w-xl p-8 text-center sm:p-10">
        <div
          className="mx-auto mb-5 grid h-20 w-20 place-items-center rounded-full"
          style={{ background: "var(--accent-soft)", color: "var(--accent)", border: "1px solid var(--accent-line)" }}
        >
          <Compass size={32} />
        </div>
        <div className="micro mb-3">Page not found</div>
        <h1 className="display mb-3 text-[clamp(2rem,4vw,3rem)]">This page drifted off the map.</h1>
        <p className="mx-auto max-w-[46ch] text-[14px] leading-relaxed text-2">
          The link may be outdated or mistyped. You can return to the national outlook or open the live risk map.
        </p>
        <div className="mt-7 flex flex-wrap items-center justify-center gap-3">
          <Link to="/" className="btn btn--primary"><Home size={15} /> Back to overview</Link>
          <Link to="/map" className="btn btn--ghost"><Map size={15} /> Open risk map</Link>
        </div>
      </Card>
    </div>
  );
}

export default NotFoundPage;
