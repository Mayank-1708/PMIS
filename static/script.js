// Navbar toggle (mobile)
const navToggle = document.getElementById("navToggle");
const navLinks = document.getElementById("navLinks");

if (navToggle && navLinks) {
  navToggle.addEventListener("click", () => {
    navLinks.classList.toggle("show");
  });

  // ✅ Close navbar menu when clicking outside
  document.addEventListener("click", (e) => {
    if (!navLinks.contains(e.target) && !navToggle.contains(e.target)) {
      navLinks.classList.remove("show");
    }
  });
}

// Password toggle
const togglePass = document.getElementById("togglePass");
const passwordInput = document.getElementById("passwordInput");

if (togglePass && passwordInput) {
  togglePass.addEventListener("click", () => {
    const isHidden = passwordInput.type === "password";
    passwordInput.type = isHidden ? "text" : "password";
    togglePass.textContent = isHidden ? "Hide" : "Show";
  });
}

// Sidebar toggle (mobile)
const sideToggle = document.getElementById("sideToggle");
const sidebar = document.getElementById("sidebar");

// ✅ Create overlay dynamically (for mobile sidebar)
let overlay = document.createElement("div");
overlay.className = "sidebar-overlay";
document.body.appendChild(overlay);

function openSidebar() {
  sidebar.classList.add("open");
  overlay.classList.add("show");
}

function closeSidebar() {
  sidebar.classList.remove("open");
  overlay.classList.remove("show");
}

if (sideToggle && sidebar) {
  sideToggle.addEventListener("click", (e) => {
    e.stopPropagation();
    if (sidebar.classList.contains("open")) {
      closeSidebar();
    } else {
      openSidebar();
    }
  });
}

// ✅ close sidebar when clicking overlay
overlay.addEventListener("click", () => {
  closeSidebar();
});

// ✅ close sidebar on pressing ESC
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeSidebar();
  }
});

// Tabs
const tabButtons = document.querySelectorAll(".side-link[data-tab]");
const tabs = document.querySelectorAll(".tab");

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");

    const target = btn.getAttribute("data-tab");
    tabs.forEach((t) => t.classList.remove("active"));

    const activeTab = document.getElementById(target);
    if (activeTab) activeTab.classList.add("active");

    // ✅ close sidebar on mobile after tab click
    closeSidebar();
  });
});

// ✅ Journey bar step update based on selected tab
function updateJourney(stepName) {
  const steps = document.querySelectorAll(".journey-step");

  // ✅ remove active from all steps
  steps.forEach((s) => s.classList.remove("active"));

  // ✅ Always keep step 1 active
  steps[0]?.classList.add("active");

  // ✅ activate based on tab
  if (stepName === "internships") steps[1]?.classList.add("active");
  if (stepName === "status") steps[2]?.classList.add("active");
}

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.getAttribute("data-tab");
    if (target === "profileTab") updateJourney("profile");
    if (target === "internshipsTab") updateJourney("internships");
    if (target === "statusTab") updateJourney("status");
  });
});

// ✅ Auto hide toast popup after 3 sec
setTimeout(() => {
  document.querySelectorAll(".toast").forEach((toast) => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(-10px)";
    setTimeout(() => toast.remove(), 400);
  });
}, 6000);
