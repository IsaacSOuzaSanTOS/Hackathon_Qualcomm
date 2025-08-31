
"use client";

import { useState } from "react";
import NavBar from "../components/NavBar";
import DevelopersModal from "../components/DevelopersModal";

export default function ClientShell({ children }: { children: React.ReactNode }) {
  const [showDevelopers, setShowDevelopers] = useState(false);

  return (
    <>
      <NavBar onClickDevelopers={() => setShowDevelopers(true)} />
      <main style={{ flex: 1 }}>{children}</main>

      {showDevelopers && (
        <DevelopersModal onClose={() => setShowDevelopers(false)} />
      )}
    </>
  );
}
