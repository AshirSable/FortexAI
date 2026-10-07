// seconds -> "5 h 12 min" / "12 min" / "3 days 4 h"
export function fmtDuration(seconds) {
  const mins = Math.max(1, Math.ceil(seconds / 60));
  if (mins < 60) return mins + ' min';
  const hours = Math.floor(mins / 60);
  if (hours < 24) return hours + ' h ' + (mins % 60) + ' min';
  return Math.floor(hours / 24) + ' days ' + (hours % 24) + ' h';
}
