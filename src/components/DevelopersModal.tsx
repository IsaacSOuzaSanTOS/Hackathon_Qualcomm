// components/DevelopersModal.tsx
"use client";
import classes from "../styles/DevelopersModal.module.css"
import Developers from "@/components/Developers";

export default function DevelopersModal({ onClose }: { onClose: () => void }) {
  return (
    <div className={classes.pai}
      onClick={onClose}>
        
      <div className={classes.modal} 
      onClick={(e) => e.stopPropagation()}>
        <Developers />
      </div>
    </div>
  );
}