// components/NavBar.tsx
"use client";
import classes from "../styles/NavBar.module.css";
import { useState } from "react";

interface NavBarProps {
  onClickDevelopers?: () => void;
}

export default function NavBar({ onClickDevelopers }: NavBarProps) {

  const [isOpen, setIsOpen] = useState(false); // Estado para controlar o menu

 
    const toggleMenu = () => {
    setIsOpen(!isOpen); 
    };


  return (
    <div className={classes.pai}>
      <div className={classes.logo}>
        <a
          href="https://www.qualcomm.com/developer/events/edge-ai-developer-hackathon/brazil"
          target="_blank"
          rel="noopener noreferrer"
        >
          <p>Edge AI Developer Hackathon</p>
        </a>
      </div>

      <div className={`${classes.central} ${isOpen ? classes.menuActive : ""}`}>
        <p onClick={onClickDevelopers}  >
          Sobre os desenvolvedores
        </p>
      </div>

      <div className={classes.final} onClick={toggleMenu}>☰</div>
    </div>
  );
}
