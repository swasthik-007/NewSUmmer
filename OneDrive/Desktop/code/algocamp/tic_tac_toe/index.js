document.addEventListener("DOMContentLoaded",()=>{
    let outer=document.getElementById("outer");
    let chance=false;//when chance is false we put o , when chance is true we put x
    let arr=Array(9).fill(undefined);

    
    outer.addEventListener("click",(e) => {
        let cell = e.target;
        let cellnumber =  cell.getAttribute("data-cell");
        if(cell.getAttribute("data-clicked")){
            return ;
            
        }
        cell.setAttribute("data-clicked","true");
        if(chance==true){
            cell.textContent="X";
            arr[cellnumber]="X";
            winningcombo("X");
        }
        else{
            cell.textContent="O";
            arr[cellnumber]="O";
            winningcombo("O");
        }   
            chance=!chance//toggle chance 
    });
    function winningcombo(char){
        let result=document.getElementById("result");
        if(arr[0]==char && arr[1]==char && arr[2]==char){
            //the f0th row is hhaving the char
            result.textContent=`${char} wins`;
        }
        else if(arr[3]==char && arr[4]==char && arr[5]==char){
            //the 1 row is hhaving the char
            result.textContent=`${char} wins`;
        }
        else if(arr[0]==char && arr[3]==char && arr[6]==char){
            //the 1 row is hhaving the char
            result.textContent=`${char} wins`;
        }
        else  if(arr[6]==char && arr[7]==char && arr[8]==char){
            //the 2 row is hhaving the char
            result.textContent=`${char} wins`;
        }
        else if(arr[1]==char && arr[4]==char && arr[7]==char){
            //the 1 col is hhaving the char
            result.textContent=`${char} wins`;
        }
        else if(arr[2]==char && arr[5]==char && arr[8]==char){
            //the 1 col is hhaving the char
            result.textContent=`${char} wins`;
        }
        else if(arr[2]==char && arr[4]==char && arr[6]==char){
            //the 1 col is hhaving the char
            result.textContent=`${char} wins`;
        }
        else if(arr[0]==char && arr[4]==char && arr[8]==char){
            //the 1 col is hhaving the char
            result.textContent=`${char} wins`;
        }

        
    }
});