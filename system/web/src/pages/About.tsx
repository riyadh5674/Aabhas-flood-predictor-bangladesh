import { ArrowRight, BookOpen, CircleHelp, Compass, ExternalLink, ShieldCheck, Waves } from "lucide-react";
import { Link } from "react-router-dom";
import { useApp } from "@/state/AppProvider";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card } from "@/components/ui/Card";

export function AboutPage() {
  const { lang } = useApp();
  return (
    <div className="nodi-info-page space-y-6">
      <PageHeader eyebrow="A guide to Aabhas" title="Understand the outlook. Feel ready to act." description="Aabhas helps people explore river-station flood risk in Bangladesh, understand what the forecast can tell them, and find sensible ways to prepare." />
      <div className="info-intro-grid">
        <Card className="info-meaning-card"><span className="info-illustration"><Waves size={33} /></span><div className="micro">What Aabhas shows</div><h2>One clear view of the river outlook.</h2><p>Explore station-level model estimates at 24, 48, and 72 hours. The map groups those station signals by division or district to help you decide where to look closer.</p><Link to="/map" className="text-link">Explore the map <ArrowRight size={14} /></Link></Card>
        <Card className="info-explainer-card"><div className="micro">Risk levels, in plain language</div><div className="risk-explainer-list">{[
          ["low", "Low", "Routine awareness. Keep up with local conditions."],
          ["moderate", "Watch", "Conditions may need attention. Check current official updates."],
          ["high", "High", "Take the signal seriously and review your household plan."],
          ["very_high", "Very high", "Prepare to act. Follow official instructions for your area."],
        ].map(([band, label, detail]) => <div className="risk-explainer-row" key={band}><span className="risk-explainer-row__dot" style={{ background: `var(--band-${band.replace("_", "-")})` }} /><strong>{label}</strong><p>{detail}</p></div>)}</div><div className="info-note"><ShieldCheck size={15} /><p>A model estimate is not a declaration that flooding will or won’t happen.</p></div></Card>
      </div>
      <Card className="info-prep-card" id="prepare"><div className="info-prep-card__heading"><span className="info-illustration"><Compass size={25} /></span><div><div className="micro">Preparedness, one step at a time</div><h2>A little planning can make a difficult day easier.</h2></div></div><div className="prep-steps"><div><span>01</span><strong>Know where to go</strong><p>Ask your local authorities about nearby shelters and safe, higher ground.</p></div><div><span>02</span><strong>Keep essentials together</strong><p>Prepare medicines, drinking water, a torch, phone charger, and important documents.</p></div><div><span>03</span><strong>Make a family plan</strong><p>Agree on a meeting point and how you’ll check in if mobile networks are busy.</p></div><div><span>04</span><strong>Follow local instructions</strong><p>Use FFWC and local authority advisories for official warnings and evacuation guidance.</p></div></div></Card>
      <div className="info-bottom-grid"><Card className="info-source-card"><div className="micro">Official information</div><h2>Stay connected to local updates.</h2><p>Aabhas is a research prototype. For official flood forecasts and warnings, follow Bangladesh’s Flood Forecasting and Warning Centre (FFWC) and your local authorities.</p><a href="https://ffwc.gov.bd/" target="_blank" rel="noreferrer">Visit FFWC <ExternalLink size={13} /></a></Card><Card className="info-source-card"><div className="micro">Data & model</div><h2>Where the outlook comes from.</h2><p>Station metadata comes from the FFWC gauge network. Risk estimates come from the project’s D16 model and its configured thresholds. If the API cannot return a forecast, Aabhas shows the error and does not replace it with synthetic values.</p><Link to="/model">How the model works <ArrowRight size={14} /></Link></Card></div>
      <div className="info-faq"><div className="micro"><CircleHelp size={13} /> A few helpful answers</div><details><summary>Why does a station show “No signal”?</summary><p>The system has no usable prediction for that station and date. Check the date selector or try again later.</p></details><details><summary>Why does the app say “Sample outlook”?</summary><p>Sample outlook appears only when you explicitly turn on Demo mode. These generated values are for exploring the interface and are not based on current river conditions.</p></details><details><summary>Can I use Aabhas as an official warning?</summary><p>No. Aabhas is a research prototype and does not issue official warnings. Follow FFWC and local authority guidance.</p></details></div>
      <div className="info-locale-note"><BookOpen size={14} />{lang === "bn" ? "বাংলা অনুবাদ ধীরে ধীরে উন্নত করা হচ্ছে।" : "Aabhas is designed for Bangladesh, with English and Bengali interface options."}</div>
    </div>
  );
}

export default AboutPage;
