import { useApp } from "@/state/AppProvider";
import icon from "@/assets/brand/aabhas-icon.svg";
import iconDark from "@/assets/brand/aabhas-icon-dark.svg";
import logo from "@/assets/brand/aabhas-logo.svg";
import logoDark from "@/assets/brand/aabhas-logo-dark.svg";

export function BrandMark({
  compact = false,
  size = 30,
}: {
  compact?: boolean;
  size?: number;
}) {
  const { theme } = useApp();
  const source = compact
    ? theme === "dark" ? iconDark : icon
    : theme === "dark" ? logoDark : logo;

  return (
    <img
      src={source}
      alt={compact ? "Aabhas" : "Aabhas — See What Comes Next"}
      className="block shrink-0 object-contain"
      style={compact ? { width: size, height: size } : { width: 154, height: 42 }}
    />
  );
}
