import classes from "../styles/Developers.module.css"

export default function Developers() {
    return(
        <div className={classes.pai}>
            <div className={classes.geral}>



                <div className={classes.dev}>
                    <h1>Antonio Iranilson Honorato</h1>
                    <a href="https://github.com/iranhonorato" target="_blank">
                        <div className={classes.github}>
                            <p>iranhonorato</p>
                            <img src="/github-mark.png"/>
                        </div>
                    </a>
                </div>



                <div className={classes.dev}>
                    <h1>Felipe Serra</h1>
                    <a href="https://github.com/Felipe-Serra-Silva" target="_blank">
                        <div className={classes.github}>
                            <p>Felipe-Serra-Silva</p>
                            <img src="/github-mark.png"/>
                        </div>
                    </a>
                </div>



                <div className={classes.dev}>
                    <h1>Gustavo Santana</h1>
                    <a href="https://github.com/YouCanCallMeGus/YouCanCallMeGus" target="_blank">
                        <div className={classes.github}>
                            <p>YouCanCallMeGus</p>
                            <img src="/github-mark.png"/>
                        </div>
                    </a>
                </div>




                <div className={classes.dev}>
                    <h1>Isaac Souza</h1>
                    <a href="https://github.com/IsaacSOuzaSanTOS" target="_blank">
                        <div className={classes.github}>
                            <p>IsaacSOuzaSanTOS</p>
                            <img src="/github-mark.png"/>
                        </div>
                    </a>
                </div>




                <div className={classes.dev}>
                    <h1>Matheus Pereira</h1>
                    <a href="https://github.com/mthperera" target="_blank">
                        <div className={classes.github}>
                            <p>mthperera</p>
                            <img src="/github-mark.png"/>
                        </div>
                    </a>
                </div>  



            </div>
        </div>
    )
}