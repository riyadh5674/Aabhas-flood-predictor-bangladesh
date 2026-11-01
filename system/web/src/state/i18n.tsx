import { createContext, useCallback, useContext, useMemo } from "react";

export type Lang = "en" | "bn";

/**
 * Curated bilingual UI strings. Long-form analytical copy stays in English
 * (the technical audience); navigation, headings, risk language and actions
 * are translated, because those are what a citizen actually needs to read.
 */
const DICT: Record<string, { en: string; bn: string }> = {
  "nav.overview": { en: "Overview", bn: "সারসংক্ষেপ" },
  "nav.map": { en: "Risk map", bn: "ঝুঁকির মানচিত্র" },
  "nav.stations": { en: "Stations", bn: "স্টেশন" },
  "nav.alerts": { en: "Alerts", bn: "সতর্কতা" },
  "nav.model": { en: "Model", bn: "মডেল" },
  "nav.about": { en: "About", bn: "পরিচিতি" },

  "common.live": { en: "Live", bn: "লাইভ" },
  "common.demo": { en: "Demo data", bn: "ডেমো ডেটা" },
  "common.offline": { en: "API offline", bn: "এপিআই বন্ধ" },
  "common.retry": { en: "Retry", bn: "আবার চেষ্টা" },
  "common.search": { en: "Search", bn: "খুঁজুন" },
  "common.date": { en: "Forecast date", bn: "পূর্বাভাসের তারিখ" },
  "common.horizon": { en: "Horizon", bn: "সময়সীমা" },
  "common.division": { en: "Division", bn: "বিভাগ" },
  "common.district": { en: "District", bn: "জেলা" },
  "common.river": { en: "River", bn: "নদী" },
  "common.station": { en: "Station", bn: "স্টেশন" },
  "common.probability": { en: "Probability", bn: "সম্ভাবনা" },
  "common.action": { en: "Action", bn: "পদক্ষেপ" },
  "common.riskBand": { en: "Risk band", bn: "ঝুঁকির মাত্রা" },
  "common.dangerLevel": { en: "Danger level", bn: "বিপদসীমা" },
  "common.export": { en: "Export CSV", bn: "CSV ডাউনলোড" },
  "common.viewAll": { en: "View all", bn: "সব দেখুন" },
  "common.loading": { en: "Loading", bn: "লোড হচ্ছে" },
  "common.noData": { en: "No data", bn: "তথ্য নেই" },

  "band.low": { en: "Low", bn: "নিম্ন" },
  "band.moderate": { en: "Moderate", bn: "মাঝারি" },
  "band.high": { en: "High", bn: "উচ্চ" },
  "band.very_high": { en: "Very high", bn: "অতি উচ্চ" },
  "band.unknown": { en: "No signal", bn: "সংকেত নেই" },

  "overview.title": { en: "National situation", bn: "জাতীয় পরিস্থিতি" },
  "overview.stationsReporting": { en: "Stations reporting", bn: "রিপোর্টিং স্টেশন" },
  "overview.maxProbability": { en: "Highest probability", bn: "সর্বোচ্চ সম্ভাবনা" },
  "overview.meanProbability": { en: "Mean probability", bn: "গড় সম্ভাবনা" },
  "overview.elevated": { en: "High or very high", bn: "উচ্চ বা অতি উচ্চ" },
  "overview.byDivision": { en: "Risk by division", bn: "বিভাগ অনুযায়ী ঝুঁকি" },
  "overview.watchlist": { en: "Watchlist", bn: "নজরতালিকা" },
  "overview.horizons": { en: "Forecast horizons", bn: "পূর্বাভাসের সময়সীমা" },

  "map.title": { en: "National risk map", bn: "জাতীয় ঝুঁকির মানচিত্র" },
  "map.legend": { en: "Legend", bn: "নির্দেশিকা" },
  "map.level": { en: "Boundary", bn: "সীমানা" },
  "map.stations": { en: "Stations", bn: "স্টেশন" },
  "map.noStation": { en: "No station coverage", bn: "কোনো স্টেশন নেই" },

  "stations.title": { en: "Station directory", bn: "স্টেশন তালিকা" },
  "stations.search": { en: "Search name, river, district…", bn: "নাম, নদী, জেলা খুঁজুন…" },
  "stations.allDivisions": { en: "All divisions", bn: "সব বিভাগ" },
  "stations.allBands": { en: "All bands", bn: "সব মাত্রা" },

  "alerts.title": { en: "Alerts", bn: "সতর্কতা" },
  "alerts.watch": { en: "Watch", bn: "নজর" },
  "alerts.warning": { en: "Warning", bn: "সতর্কবার্তা" },
  "alerts.high": { en: "High-risk", bn: "উচ্চ ঝুঁকি" },
  "alerts.none": { en: "No elevated alerts", bn: "কোনো সতর্কতা নেই" },

  "hero.eyebrow": { en: "Flood risk intelligence · Bangladesh", bn: "বন্যা ঝুঁকি · বাংলাদেশ" },
  "hero.title": { en: "See what comes next.", bn: "যা আসছে, তা দেখুন।" },
  "home.titleA": { en: "Know your river.", bn: "নদীকে চিনুন।" },
  "home.titleB": { en: "Plan ahead.", bn: "আগেই প্রস্তুত থাকুন।" },
  "home.description": { en: "Explore station-level flood risk estimates across Bangladesh. Start with the map or find a river station near you.", bn: "বাংলাদেশজুড়ে স্টেশনভিত্তিক বন্যার ঝুঁকির পূর্বাভাস দেখুন। মানচিত্র থেকে শুরু করুন, অথবা কাছের নদী স্টেশন খুঁজুন।" },
  "home.openMap": { en: "Explore the risk map", bn: "ঝুঁকির মানচিত্র দেখুন" },
  "home.findStation": { en: "Find a station", bn: "স্টেশন খুঁজুন" },
  "home.startRegion": { en: "Start with your region", bn: "আপনার এলাকার পূর্বাভাস" },
  "home.regionTitle": { en: "How is the river outlook changing?", bn: "আপনার এলাকার নদীর পরিস্থিতি কেমন?" },
  "home.regionHelp": { en: "Select a division to see its stations and forecast details.", bn: "স্টেশন ও পূর্বাভাসের বিস্তারিত দেখতে একটি বিভাগ বেছে নিন।" },
  "home.allRegions": { en: "See all divisions", bn: "সব বিভাগ দেখুন" },

  "disclaimer.short": {
    en: "Research prototype. Not an official warning service.",
    bn: "গবেষণা প্রকল্প। এটি সরকারি সতর্কতা ব্যবস্থা নয়।",
  },
};

interface I18nValue {
  lang: Lang;
  setLang: (lang: Lang) => void;
  t: (key: string) => string;
}

const I18nContext = createContext<I18nValue | null>(null);

export function I18nProvider({
  lang,
  setLang,
  children,
}: {
  lang: Lang;
  setLang: (lang: Lang) => void;
  children: React.ReactNode;
}) {
  const t = useCallback(
    (key: string) => {
      const entry = DICT[key];
      if (!entry) return key;
      return entry[lang] ?? entry.en;
    },
    [lang],
  );

  const value = useMemo(() => ({ lang, setLang, t }), [lang, setLang, t]);
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nValue {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n must be used inside I18nProvider");
  return ctx;
}
