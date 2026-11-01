import { ExternalLink, Github, ShieldAlert } from "lucide-react";
import { Link } from "react-router-dom";
import { BrandMark } from "./BrandMark";

const COLUMNS = [
  {
    title: "Product",
    links: [
      { to: "/", label: "Overview" },
      { to: "/map", label: "Risk map" },
      { to: "/stations", label: "Stations" },
      { to: "/alerts", label: "Alerts" },
    ],
  },
  {
    title: "System",
    links: [
      { to: "/model", label: "Model card" },
      { to: "/about", label: "About & FAQ" },
      { to: "/model#thresholds", label: "Thresholds" },
      { to: "/about#data", label: "Data sources" },
    ],
  },
];

export function Footer() {
  return (
    <footer className="mt-16 border-t px-4 pb-24 pt-10 no-print sm:px-6 lg:pb-10" style={{ borderColor: "var(--border)" }}>
      <div className="mx-auto max-w-[1500px]">
        <div className="grid gap-10 md:grid-cols-[1.4fr_1fr_1fr_1.2fr]">
          <div>
            <BrandMark size={32} />
            <p className="mt-4 max-w-[34ch] text-[13px] leading-relaxed text-2">
              A clear view of river-station flood outlooks across Bangladesh, with practical guidance to help people prepare.
            </p>
            <div className="mt-4 flex items-center gap-2">
              <span className="chip" title="Frozen model artefact">
                D16 · FULL_V2
              </span>
              <span className="chip">Research preview</span>
            </div>
          </div>

          {COLUMNS.map((column) => (
            <div key={column.title}>
              <div className="micro mb-3">{column.title}</div>
              <ul className="space-y-2">
                {column.links.map((link) => (
                  <li key={link.to + link.label}>
                    <Link
                      to={link.to}
                      className="text-[13px] text-2 transition-colors hover:text-[var(--accent)]"
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}

          <div>
            <div className="micro mb-3">Sources</div>
            <ul className="space-y-2 text-[13px] text-2">
              <li>
                <a
                  href="https://ffwc.gov.bd/"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 transition-colors hover:text-[var(--accent)]"
                >
                  FFWC gauge network <ExternalLink size={11} />
                </a>
              </li>
              <li>
                <a
                  href="https://www.geoboundaries.org/"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 transition-colors hover:text-[var(--accent)]"
                >
                  geoBoundaries admin layers <ExternalLink size={11} />
                </a>
              </li>
              <li>
                <a
                  href="https://www.bwdb.gov.bd/"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 transition-colors hover:text-[var(--accent)]"
                >
                  BWDB hydrological data <ExternalLink size={11} />
                </a>
              </li>
            </ul>
            <div className="mt-4 flex items-center gap-2">
              <Github size={13} className="faint" />
              <span className="text-[11.5px] faint">Research build · not a public service</span>
            </div>
          </div>
        </div>

        <div
          className="mt-9 flex flex-col gap-3 border-t pt-5 text-[11.5px] faint sm:flex-row sm:items-center sm:justify-between"
          style={{ borderColor: "var(--hairline)" }}
        >
          <div className="flex items-center gap-2">
            <ShieldAlert size={13} />
            <span>
              Aabhas is a research prototype, not an official warning service. Always follow FFWC and local authority advisories.
            </span>
          </div>
          <div className="num whitespace-nowrap">© 2026 Aabhas · Made for Bangladesh</div>
        </div>
      </div>
    </footer>
  );
}
