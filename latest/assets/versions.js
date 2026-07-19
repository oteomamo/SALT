/* The version selector is a shortcut to the version history, not a
   portal to stale snapshots: every entry routes to the changelog. */
document.addEventListener(
  "click",
  function (ev) {
    var link = ev.target.closest ? ev.target.closest("a.md-version__link") : null;
    if (!link) return;
    ev.preventDefault();
    ev.stopPropagation();
    var m = window.location.pathname.match(/^(.*?\/)(?:latest|\d[\w.\-]*)\//);
    window.location.href = (m ? m[1] : "/") + "latest/changelog/";
  },
  true
);
