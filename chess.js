const gameBoard = document.querySelector("#gameboard");
const playerDisplay = document.querySelector("#player");
const infoDisplay = document.querySelector("#info-display");

let playerGo='black';
playerDisplay.textContent=playerGo.toUpperCase();

const width = 8; // 8 by 8 board

// all the pieces 

// const king = '<div class="piece" id="king"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512"><!--!Font Awesome Free 6.5.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path d="M224 0c17.7 0 32 14.3 32 32V48h16c17.7 0 32 14.3 32 32s-14.3 32-32 32H256v48H408c22.1 0 40 17.9 40 40c0 5.3-1 10.5-3.1 15.4L368 400H80L3.1 215.4C1 210.5 0 205.3 0 200c0-22.1 17.9-40 40-40H192V112H176c-17.7 0-32-14.3-32-32s14.3-32 32-32h16V32c0-17.7 14.3-32 32-32zM38.6 473.4L80 432H368l41.4 41.4c4.2 4.2 6.6 10 6.6 16c0 12.5-10.1 22.6-22.6 22.6H54.6C42.1 512 32 501.9 32 489.4c0-6 2.4-11.8 6.6-16z"/></svg></div>'
// const queen = `<div class="piece" id="queen"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><!--!Font Awesome Free 6.5.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path d="M256 0a56 56 0 1 1 0 112A56 56 0 1 1 256 0zM134.1 143.8c3.3-13 15-23.8 30.2-23.8c12.3 0 22.6 7.2 27.7 17c12 23.2 36.2 39 64 39s52-15.8 64-39c5.1-9.8 15.4-17 27.7-17c15.3 0 27 10.8 30.2 23.8c7 27.8 32.2 48.3 62.1 48.3c10.8 0 21-2.7 29.8-7.4c8.4-4.4 18.9-4.5 27.6 .9c13 8 17.1 25 9.2 38L399.7 400H80L3.1 215.4C1 210.5 0 205.3 0 200c0-22.1 17.9-40 40-40H192V112H176c-17.7 0-32-14.3-32-32s14.3-32 32-32h16V32c0-17.7 14.3-32 32-32zM256 224l0 0 0 0h0zM112 432H400l41.4 41.4c4.2 4.2 6.6 10 6.6 16c0 12.5-10.1 22.6-22.6 22.6H86.6C74.1 512 64 501.9 64 489.4c0-6 2.4-11.8 6.6-16L112 432z"/></svg></div>`
// const rook = `<div class="piece" id="rook"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512"><!--!Font Awesome Free 6.5.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path d="M32 192V48c0-8.8 7.2-16 16-16h64c8.8 0 16 7.2 16 16V88c0 4.4 3.6 8 8 8h32c4.4 0 8-3.6 8-8V48c0-8.8 7.2-16 16-16h64c8.8 0 16 7.2 16 16V88c0 4.4 3.6 8 8 8h32c4.4 0 8-3.6 8-8V48c0-8.8 7.2-16 16-16h64c8.8 0 16 7.2 16 16V192c0 10.1-4.7 19.6-12.8 25.6L352 256l16 144H80L96 256 44.8 217.6C36.7 211.6 32 202.1 32 192zm176 96h32c8.8 0 16-7.2 16-16V224c0-17.7-14.3-32-32-32s-32 14.3-32 32v48c0 8.8 7.2 16 16 16zM22.6 473.4L64 432H384l41.4 41.4c4.2 4.2 6.6 10 6.6 16c0 12.5-10.1 22.6-22.6 22.6H38.6C26.1 512 16 501.9 16 489.4c0-6 2.4-11.8 6.6-16z"/></svg></div>`
// const bishop = `<div class="piece" id="bishop"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 512"><!--!Font Awesome Free 6.5.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path d="M128 0C110.3 0 96 14.3 96 32c0 16.1 11.9 29.4 27.4 31.7C78.4 106.8 8 190 8 288c0 47.4 30.8 72.3 56 84.7V400H256V372.7c25.2-12.5 56-37.4 56-84.7c0-37.3-10.2-72.4-25.3-104.1l-99.4 99.4c-6.2 6.2-16.4 6.2-22.6 0s-6.2-16.4 0-22.6L270.8 154.6c-23.2-38.1-51.8-69.5-74.2-90.9C212.1 61.4 224 48.1 224 32c0-17.7-14.3-32-32-32H128zM48 432L6.6 473.4c-4.2 4.2-6.6 10-6.6 16C0 501.9 10.1 512 22.6 512H297.4c12.5 0 22.6-10.1 22.6-22.6c0-6-2.4-11.8-6.6-16L272 432H48z"/></svg></div>`
// const pawn = `<div class="piece" id="pawn"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 512"><!--!Font Awesome Free 6.5.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path d="M215.5 224c29.2-18.4 48.5-50.9 48.5-88c0-57.4-46.6-104-104-104S56 78.6 56 136c0 37.1 19.4 69.6 48.5 88H96c-17.7 0-32 14.3-32 32c0 16.5 12.5 30 28.5 31.8L80 400H240L227.5 287.8c16-1.8 28.5-15.3 28.5-31.8c0-17.7-14.3-32-32-32h-8.5zM22.6 473.4c-4.2 4.2-6.6 10-6.6 16C16 501.9 26.1 512 38.6 512H281.4c12.5 0 22.6-10.1 22.6-22.6c0-6-2.4-11.8-6.6-16L256 432H64L22.6 473.4z"/></svg></div>`
// const knight = `<div class="piece" id="knight"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512"><!--!Font Awesome Free 6.5.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path d="M96 48L82.7 61.3C70.7 73.3 64 89.5 64 106.5V238.9c0 10.7 5.3 20.7 14.2 26.6l10.6 7c14.3 9.6 32.7 10.7 48.1 3l3.2-1.6c2.6-1.3 5-2.8 7.3-4.5l49.4-37c6.6-5 15.7-5 22.3 0c10.2 7.7 9.9 23.1-.7 30.3L90.4 350C73.9 361.3 64 380 64 400H384l28.9-159c2.1-11.3 3.1-22.8 3.1-34.3V192C416 86 330 0 224 0H83.8C72.9 0 64 8.9 64 19.8c0 7.5 4.2 14.3 10.9 17.7L96 48zm24 68a20 20 0 1 1 40 0 20 20 0 1 1 -40 0zM22.6 473.4c-4.2 4.2-6.6 10-6.6 16C16 501.9 26.1 512 38.6 512H409.4c12.5 0 22.6-10.1 22.6-22.6c0-6-2.4-11.8-6.6-16L384 432H64L22.6 473.4z"/></svg></div>`

//all the pieces 


const startPieces = [
    rook, knight, bishop, queen, king, bishop, knight, rook,
    pawn, pawn, pawn, pawn, pawn, pawn, pawn, pawn,
    '', '', '', '', '', '', '', '',
    '', '', '', '', '', '', '', '',
    '', '', '', '', '', '', '', '',
    '', '', '', '', '', '', '', '',
    pawn, pawn, pawn, pawn, pawn, pawn, pawn, pawn,
    rook, knight, bishop, queen, king, bishop, knight, rook
];

function createBoard() {
    startPieces.forEach((startPiece, i) => {
        const square = document.createElement('div'); //we are making a div tag for all the squares 
        square.classList.add('square'); // adding square class to the divs 
        square.innerHTML = startPiece; // we are naming the squares so we can put the images ,svg 

        square.firstChild?.setAttribute('draggable',true); // if the square has a first child then make draggable as true 

        //if it has a first child then it is draggable else it isnt specified


        
        square.setAttribute('square-id', i); // we are putting the square id as the index of it so 0 to 63


        const row = Math.floor((63 - i )/8)+1; //this checks for the row the element is in , which is out of 8 rows so it is divising by 8 and adding 1 to get a value from 1 to 8 

        if(row % 2===0){ // if the row is even 
            square.classList.add(i % 2===0 ? "beige" : "brown"); // then we check if the index is even , if index also is even then it is beige or else it is black 
        }
        else{
            square.classList.add(i % 2===0 ? "brown" : "beige");   // of the row number is odd then we check if the index is even , if even then it is brown or else it is beige 
        }
        
        if(i<=15){ // first 16
            square.firstChild.firstChild.classList.add("black");  // for the first 16 blocks we add class black as they are the black pieces 
        }
        else if(i>=48){// last 16
            square.firstChild.firstChild.classList.add("white"); // white pieces last 16 pieces 
        }

        gameBoard.append(square); // we append the square to the gameboard one by one after going through all these properties 
    });
}

createBoard();  // we are creating the board 


const allSquares = document.querySelectorAll('.square');  // allsquares connects to all the squares on the board

allSquares.forEach(square =>{ // for each square we are checking id any of the following happens as it is an addeventlistener so it listens to the event that will take place 

    square.addEventListener('dragstart',dragStart); // as soon as we start dragging the input is taken 
    square.addEventListener('dragover',dragOver); // as soon as we drag over something the functionn dragover is called 
    square.addEventListener('drop',dragDrop); // as soon as we drop the piece
    
})


let startPositionId ;
let draggedElement;


function dragStart(e){
    startPositionId = e.target.parentNode.getAttribute('square-id'); // the square from which the drag has started which is the square which is dragged // gets the square where the piece is moved from 
    draggedElement = e.target; // making it easy as the whole info of the dragged element is stored here 

}
function dragOver(e){
    e.preventDefault(); // when we are dragging over the default drag shouldnt happen // didnt understand


}

function dragDrop(e){
    // we want to drag and drop into the square itself and not the svg or something 
    e.stopPropagation();

    // console.log('playerGo',playerGo);

    console.log('where the piece is dropped',e.target);
    const correctGo = draggedElement.firstChild.classList.contains(playerGo) ; 
   
    // this stores if the player who has mooved it is his turn or not because only when it is his turn he will get to move and stuff and when it is the opponents move 
    //e.target is what we are dragging the pience into

    const taken = e.target.classList.contains('piece'); //taken has all the squares that contains a piece 


    // we check the validity of the square into the function checkifvalid 
    const valid = checkIfvalid(e.target);

    //opponent go is opposite of player go 
    const opponentGo = playerGo ==="white" ?'black' : 'white'; // if playerGo is white then the opponent is black and if not then oponent is white itself 
    
    // console.log("opponent GO is ",opponentGo);

    const takenByOpponent = e.target.firstChild ?.classList.contains(opponentGo);  // this is checked everytime if the opponent takes the element then this happens 

    
    //correctgo is the time when the move is made for the correct piece and not the opponent piece

    if(correctGo){
        //means we are the correct player check if it is taken by opponent and a valid move we want to replace the element
            //to check if it is taken by opponent and is a valid move
        if(takenByOpponent && valid){
           // if taken by opponent and a valid move , then we replace the element currently in the parent no by the dragged element .... 
           // .... and we remove whatever what there b4 it and we call the change player function also 
            e.target.parentNode.append(draggedElement);

            e.target.remove(); //remove whatever else is in there 

            
            changePlayer(); // next move 
            checkforWin();
            return;

        }
        //then check this 

        if(taken && !takenByOpponent){
            // if taken but not taken by element which means it is the same class which is either black or white 
            // then we cannot replace and show error statement for 2 seconds
            infoDisplay.textContent = " YOU CANNOT GO HERE ";
            setTimeout(() => infoDisplay.textContent ="",2000);
            //to remove the text in 2 seocnds 
            return;
        }

        if (valid){
            // if valid and no other problem is caused then we just append the draggredelement and remove the contents previously kept 

            e.target.append(draggedElement);
            
            changePlayer();
            checkforWin();
            return;

        }
        else if(!valid){
            const piece = draggedElement.id;
            const startId= Number(startPositionId)
            switch(piece){
                case 'rook':
                    infoDisplay.textContent = "Rook - Moves any number of squares horizontally or vertically.";
                    break;
                case 'pawn':
                    const starterRow=[8,9,10,11,12,13,14,15];
                    if(starterRow.includes(startId)){
                        infoDisplay.textContent = "Pawn - Moves one square forward, but on its first move, it can move two squares forward. It captures diagonally one square forward .";
                   
                    }
                    else{
                        infoDisplay.textContent = "Pawn - Moves one square forward, but on its first move, it can move two squares forward. It captures diagonally one square forward and the pawn cannot jump over other pieces";
                    }
                    break;
                case 'knight':
                    infoDisplay.textContent = "Knight - Moves in an `L-shape,` two squares in a straight direction, and then one square perpendicular to that. ";
                    break;
                case 'bishop':
                    infoDisplay.textContent = "Bishop - Moves any number of squares diagonally like SW , SE ,NW AND NE , the bishop cannot jump over other pieces ";
                    break;
                case 'king':
                    infoDisplay.textContent = "King - Moves one square in any direction. ";
                    break;
                case 'queen':
                    infoDisplay.textContent = "Queen - Moves any number of squares diagonally, horizontally, or vertically.";
                    break;
                default:
            }
        }
    }
    else if(!correctGo){
        const opponentGo = playerGo ==="white" ?'black' : 'white'; 
        infoDisplay.textContent = "WAIT FOR YOUR TURN " + opponentGo.toUpperCase() ;
        setTimeout(() => infoDisplay.textContent ="",2000);
        return;
    }

    // e.target.parentNode.append(draggedElement);  we move the element which we have chosen into the square which we are dropping it into 
   
    // Append the draggedElement to the target parent
    
    // e.target.parentNode.append(draggedElement);
    // e.target.remove();
    
    // e.target.append(draggedElement);

}


function checkIfvalid(target){

    // console.log(target);
    // if square id is not specified in the target then we check if the parent node has the squareid attribute
    const targetId = Number(target.getAttribute('square-id'))||Number(target.parentNode.getAttribute('square-id')); 

    const startId= Number(startPositionId);
     // start id is the starting position

    const piece = draggedElement.id;
    //the piece is the piece placed on the chessboard which is pawn and stuff 

    console.log('target ID',targetId); // ending position
    console.log('Start ID',startId); // starting position
    console.log('piece',piece); // which piece i am moving 

    
    switch(piece){
        case 'pawn':
            const starterRow=[8,9,10,11,12,13,14,15];
            if(starterRow.includes(startId) && startId + width * 2 === targetId ||
             startId + width === targetId || 
             startId +width+ 1 === targetId && document.querySelector(`[square-id="${startId+width + 1 }"]`).firstChild ||
             startId + width - 1 === targetId && document.querySelector(`[square-id="${startId+width - 1 }"]`).firstChild
            ){
                return true ;
            }
        break;

        case 'knight':
            if (
                startId + (2*width) +1 === targetId || //document.querySelector(`[square-id="${startId+(2*width) + 1 }"]`).firstChild ||
                startId + (2*width) -1 === targetId || //document.querySelector(`[square-id="${startId+(2*width) - 1 }"]`).firstChild ||
                startId +  width  + 2 === targetId || //document.querySelector(`[square-id="${startId+(width) + 2 }"]`).firstChild ||
                startId +  width  - 2 === targetId || //document.querySelector(`[square-id="${startId+(width) - 2 }"]`).firstChild ||
                startId - (2*width) +1 === targetId || //document.querySelector(`[square-id="${startId-(2*width) + 1 }"]`).firstChild ||
                startId - (2*width) -1 === targetId || //document.querySelector(`[square-id="${startId-(2*width) - 1 }"]`).firstChild ||
                startId -  width  + 2 === targetId || //document.querySelector(`[square-id="${startId-(width) + 2 }"]`).firstChild ||
                startId -  width  - 2 === targetId  // || document.querySelector(`[square-id="${startId-(width) - 2 }"]`).firstChild 
            ){
                return true;
            }
        break ;

        case 'rook':
            if(
            startId + (1*width) === targetId ||
            startId + (2*width) === targetId && !document.querySelector(`[square-id="${startId+(width) }"]`).firstChild ||
            startId + (3*width) === targetId && !document.querySelector(`[square-id="${startId+(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2*width) }"]`).firstChild ||
            startId + (4*width) === targetId && !document.querySelector(`[square-id="${startId+(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(3*width) }"]`).firstChild ||
            startId + (5*width) === targetId && !document.querySelector(`[square-id="${startId+(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(3*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(4*width) }"]`).firstChild ||
            startId + (6*width) === targetId && !document.querySelector(`[square-id="${startId+(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(3*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(4*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(5*width) }"]`).firstChild ||
            startId + (7*width) === targetId && !document.querySelector(`[square-id="${startId+(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(3*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(4*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(5*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(6*width) }"]`).firstChild 
            ||
            startId - (1*width) === targetId ||
            startId - (2*width) === targetId && !document.querySelector(`[square-id="${startId-(width) }"]`).firstChild ||
            startId - (3*width) === targetId && !document.querySelector(`[square-id="${startId-(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2*width) }"]`).firstChild ||
            startId - (4*width) === targetId && !document.querySelector(`[square-id="${startId-(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(3*width) }"]`).firstChild ||
            startId - (5*width) === targetId && !document.querySelector(`[square-id="${startId-(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(3*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(4*width) }"]`).firstChild ||
            startId - (6*width) === targetId && !document.querySelector(`[square-id="${startId-(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(3*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(4*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(5*width) }"]`).firstChild ||
            startId - (7*width) === targetId && !document.querySelector(`[square-id="${startId-(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(3*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(4*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(5*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(6*width) }"]`).firstChild 
            ||
            startId +  (1) === targetId ||
            startId +  (2) === targetId && !document.querySelector(`[square-id="${startId+(1) }"]`).firstChild ||
            startId +  (3) === targetId && !document.querySelector(`[square-id="${startId+(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2) }"]`).firstChild ||
            startId +  (4) === targetId && !document.querySelector(`[square-id="${startId+(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(3) }"]`).firstChild ||
            startId +  (5) === targetId && !document.querySelector(`[square-id="${startId+(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(3) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(4) }"]`).firstChild ||
            startId +  (6) === targetId && !document.querySelector(`[square-id="${startId+(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(3) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(4) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(5) }"]`).firstChild ||
            startId +  (7) === targetId && !document.querySelector(`[square-id="${startId+(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(3) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(4) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(5) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(6) }"]`).firstChild 
            ||
            startId -  (1)  === targetId ||
            startId -  (2)  === targetId && !document.querySelector(`[square-id="${startId-(1) }"]`).firstChild ||
            startId -  (3)  === targetId && !document.querySelector(`[square-id="${startId-(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2) }"]`).firstChild ||
            startId -  (4)  === targetId && !document.querySelector(`[square-id="${startId-(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(3) }"]`).firstChild ||
            startId -  (5)  === targetId && !document.querySelector(`[square-id="${startId-(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(3) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(4) }"]`).firstChild ||
            startId -  (6)  === targetId && !document.querySelector(`[square-id="${startId-(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(3) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(4) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(5) }"]`).firstChild ||
            startId -  (7)  === targetId && !document.querySelector(`[square-id="${startId-(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(3) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(4) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(5) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(6) }"]`).firstChild 
            
        ){      
            if(row % 2===0){ // if the row is even 
                square.classList.add(i % 2===0 ? "beige50" : "brown50"); // then we check if the index is even , if index also is even then it is beige or else it is black 
                square.classList.remove(i % 2===0 ? "beige" : "brown");
            }
            else{
                square.classList.add(i % 2===0 ? "brown50" : "beige50");   // of the row number is odd then we check if the index is even , if even then it is brown or else it is beige 
                square.classList.remove(i % 2===0 ? "brown" : "beige");
            }
                return true;
        }


        break ;

        case 'bishop':
            if(// all moves in all directions so 7 into 4 which is 28 moves in ++,+-,-+ and --
            startId + width + 1 === targetId  ||
            startId + width - 1 === targetId ||
            startId - width + 1 === targetId ||
            startId - width - 1 === targetId ||
            startId + (2*width) + 2 === targetId &&  !document.querySelector(`[square-id="${startId+(width)+1}"]`).firstChild ||
            startId + (2*width) - 2 === targetId &&  !document.querySelector(`[square-id="${startId+(width)-1}"]`).firstChild ||
            startId - (2*width) + 2 === targetId &&  !document.querySelector(`[square-id="${startId-(width)+1}"]`).firstChild ||
            startId - (2*width) - 2 === targetId &&  !document.querySelector(`[square-id="${startId-(width)-1}"]`).firstChild ||
            startId + (3*width) + 3 === targetId &&  !document.querySelector(`[square-id="${startId+(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 2)+2}"]`).firstChild ||
            startId + (3*width) - 3 === targetId &&  !document.querySelector(`[square-id="${startId+(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 2)-2}"]`).firstChild ||
            startId - (3*width) + 3 === targetId &&  !document.querySelector(`[square-id="${startId-(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 2)+2}"]`).firstChild ||
            startId - (3*width) - 3 === targetId &&  !document.querySelector(`[square-id="${startId-(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 2)-2}"]`).firstChild ||
            startId + (4*width) + 4 === targetId &&  !document.querySelector(`[square-id="${startId+(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 2)+2}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 3)+3}"]`).firstChild ||
            startId + (4*width) - 4 === targetId &&  !document.querySelector(`[square-id="${startId+(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 2)-2}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 3)-3}"]`).firstChild ||
            startId - (4*width) + 4 === targetId &&  !document.querySelector(`[square-id="${startId-(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 2)+2}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 3)+3}"]`).firstChild ||
            startId - (4*width) - 4 === targetId &&  !document.querySelector(`[square-id="${startId-(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 2)-2}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 3)-3}"]`).firstChild ||
            startId + (5*width) + 5 === targetId &&  !document.querySelector(`[square-id="${startId+(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 2)+2}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 3)+3}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 4)+4}"]`).firstChild ||
            startId + (5*width) - 5 === targetId &&  !document.querySelector(`[square-id="${startId+(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 2)-2}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 3)-3}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 4)-4}"]`).firstChild ||
            startId - (5*width) + 5 === targetId &&  !document.querySelector(`[square-id="${startId-(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 2)+2}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 3)+3}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 4)+4}"]`).firstChild ||
            startId - (5*width) - 5 === targetId &&  !document.querySelector(`[square-id="${startId-(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 2)-2}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 3)-3}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 4)-4}"]`).firstChild ||
            startId + (6*width) + 6 === targetId &&  !document.querySelector(`[square-id="${startId+(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 2)+2}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 3)+3}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 4)+4}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 5)+5}"]`).firstChild ||
            startId + (6*width) - 6 === targetId &&  !document.querySelector(`[square-id="${startId+(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 2)-2}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 3)-3}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 4)-4}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 5)-5}"]`).firstChild ||
            startId - (6*width) + 6 === targetId &&  !document.querySelector(`[square-id="${startId-(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 2)+2}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 3)+3}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 4)+4}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 5)+5}"]`).firstChild ||
            startId - (6*width) - 6 === targetId &&  !document.querySelector(`[square-id="${startId-(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 2)-2}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 3)-3}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 4)-4}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 5)-5}"]`).firstChild ||
            startId + (7*width) + 7 === targetId &&  !document.querySelector(`[square-id="${startId+(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 2)+2}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 3)+3}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 4)+4}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 5)+5}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 6)+6}"]`).firstChild ||
            startId + (7*width) - 7 === targetId &&  !document.querySelector(`[square-id="${startId+(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 2)-2}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 3)-3}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 4)-4}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 5)-5}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 6)-6}"]`).firstChild ||
            startId - (7*width) + 7 === targetId &&  !document.querySelector(`[square-id="${startId-(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 2)+2}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 3)+3}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 4)+4}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 5)+5}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 6)+6}"]`).firstChild ||
            startId - (7*width) - 7 === targetId &&  !document.querySelector(`[square-id="${startId-(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 2)-2}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 3)-3}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 4)-4}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 5)-5}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 6)-6}"]`).firstChild 
            ){
                return true;
            }
            break;

        case 'queen':
            if(
                
            startId + width + 1 === targetId ||
            startId + width - 1 === targetId ||
            startId - width + 1 === targetId ||
            startId - width - 1 === targetId 
            ||
            startId + (2*width) + 2 === targetId && !document.querySelector(`[square-id="${startId+(width)+1}"]`).firstChild ||
            startId + (2*width) - 2 === targetId && !document.querySelector(`[square-id="${startId+(width)-1}"]`).firstChild ||
            startId - (2*width) + 2 === targetId && !document.querySelector(`[square-id="${startId-(width)+1}"]`).firstChild ||
            startId - (2*width) - 2 === targetId && !document.querySelector(`[square-id="${startId-(width)-1}"]`).firstChild 
            ||
            startId + (3*width) + 3 === targetId && !document.querySelector(`[square-id="${startId+(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width*2)+2}"]`).firstChild ||
            startId + (3*width) - 3 === targetId && !document.querySelector(`[square-id="${startId+(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width*2)-2}"]`).firstChild ||
            startId - (3*width) + 3 === targetId && !document.querySelector(`[square-id="${startId-(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width*2)+2}"]`).firstChild ||
            startId - (3*width) - 3 === targetId && !document.querySelector(`[square-id="${startId-(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width*2)-2}"]`).firstChild 
            ||
            startId + (4*width) + 4 === targetId && !document.querySelector(`[square-id="${startId+(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width*2)+2}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 3)+3}"]`).firstChild ||
            startId + (4*width) - 4 === targetId && !document.querySelector(`[square-id="${startId+(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width*2)-2}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 3)-3}"]`).firstChild ||
            startId - (4*width) + 4 === targetId && !document.querySelector(`[square-id="${startId-(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width*2)+2}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 3)+3}"]`).firstChild ||
            startId - (4*width) - 4 === targetId && !document.querySelector(`[square-id="${startId-(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width*2)-2}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 3)-3}"]`).firstChild 
            ||
            startId + (5*width) + 5 === targetId && !document.querySelector(`[square-id="${startId+(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width*2)+2}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 3)+3}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 4)+4}"]`).firstChild ||
            startId + (5*width) - 5 === targetId && !document.querySelector(`[square-id="${startId+(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width*2)-2}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 3)-3}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 4)-4}"]`).firstChild ||
            startId - (5*width) + 5 === targetId && !document.querySelector(`[square-id="${startId-(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width*2)+2}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 3)+3}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 4)+4}"]`).firstChild ||
            startId - (5*width) - 5 === targetId && !document.querySelector(`[square-id="${startId-(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width*2)-2}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 3)-3}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 4)-4}"]`).firstChild 
            ||
            startId + (6*width) + 6 === targetId && !document.querySelector(`[square-id="${startId+(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width*2)+2}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 3)+3}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 4)+4}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 5)+5}"]`).firstChild ||
            startId + (6*width) - 6 === targetId && !document.querySelector(`[square-id="${startId+(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width*2)-2}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 3)-3}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 4)-4}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 5)-5}"]`).firstChild ||
            startId - (6*width) + 6 === targetId && !document.querySelector(`[square-id="${startId-(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width*2)+2}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 3)+3}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 4)+4}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 5)+5}"]`).firstChild ||
            startId - (6*width) - 6 === targetId && !document.querySelector(`[square-id="${startId-(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width*2)-2}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 3)-3}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 4)-4}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 5)-5}"]`).firstChild 
            ||
            startId + (7*width) + 7 === targetId && !document.querySelector(`[square-id="${startId+(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width*2)+2}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 3)+3}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 4)+4}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 5)+5}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 6)+6}"]`).firstChild ||
            startId + (7*width) - 7 === targetId && !document.querySelector(`[square-id="${startId+(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width*2)-2}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 3)-3}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 4)-4}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 5)-5}"]`).firstChild && !document.querySelector(`[square-id="${startId+(width* 6)-6}"]`).firstChild ||
            startId - (7*width) + 7 === targetId && !document.querySelector(`[square-id="${startId-(width)+1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width*2)+2}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 3)+3}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 4)+4}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 5)+5}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 6)+6}"]`).firstChild ||
            startId - (7*width) - 7 === targetId && !document.querySelector(`[square-id="${startId-(width)-1}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width*2)-2}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 3)-3}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 4)-4}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 5)-5}"]`).firstChild && !document.querySelector(`[square-id="${startId-(width* 6)-6}"]`).firstChild 
            ||
            startId + (1*width) === targetId ||
            startId + (2*width) === targetId && !document.querySelector(`[square-id="${startId+(width) }"]`).firstChild ||
            startId + (3*width) === targetId && !document.querySelector(`[square-id="${startId+(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2*width) }"]`).firstChild ||
            startId + (4*width) === targetId && !document.querySelector(`[square-id="${startId+(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(3*width) }"]`).firstChild ||
            startId + (5*width) === targetId && !document.querySelector(`[square-id="${startId+(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(3*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(4*width) }"]`).firstChild ||
            startId + (6*width) === targetId && !document.querySelector(`[square-id="${startId+(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(3*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(4*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(5*width) }"]`).firstChild ||
            startId + (7*width) === targetId && !document.querySelector(`[square-id="${startId+(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(3*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(4*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(5*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(6*width) }"]`).firstChild 
            ||
            startId - (1*width) === targetId ||
            startId - (2*width) === targetId && !document.querySelector(`[square-id="${startId-(width) }"]`).firstChild ||
            startId - (3*width) === targetId && !document.querySelector(`[square-id="${startId-(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2*width) }"]`).firstChild ||
            startId - (4*width) === targetId && !document.querySelector(`[square-id="${startId-(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(3*width) }"]`).firstChild ||
            startId - (5*width) === targetId && !document.querySelector(`[square-id="${startId-(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(3*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(4*width) }"]`).firstChild ||
            startId - (6*width) === targetId && !document.querySelector(`[square-id="${startId-(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(3*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(4*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(5*width) }"]`).firstChild ||
            startId - (7*width) === targetId && !document.querySelector(`[square-id="${startId-(width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(3*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(4*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(5*width) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(6*width) }"]`).firstChild 
            ||
            startId +  (1) === targetId ||
            startId +  (2) === targetId && !document.querySelector(`[square-id="${startId+(1) }"]`).firstChild ||
            startId +  (3) === targetId && !document.querySelector(`[square-id="${startId+(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2) }"]`).firstChild ||
            startId +  (4) === targetId && !document.querySelector(`[square-id="${startId+(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(3) }"]`).firstChild ||
            startId +  (5) === targetId && !document.querySelector(`[square-id="${startId+(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(3) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(4) }"]`).firstChild ||
            startId +  (6) === targetId && !document.querySelector(`[square-id="${startId+(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(3) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(4) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(5) }"]`).firstChild ||
            startId +  (7) === targetId && !document.querySelector(`[square-id="${startId+(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(2) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(3) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(4) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(5) }"]`).firstChild && !document.querySelector(`[square-id="${startId+(6) }"]`).firstChild 
            ||
            startId -  ( 1 )  === targetId ||
            startId -  (2*1)  === targetId && !document.querySelector(`[square-id="${startId-(1) }"]`).firstChild ||
            startId -  (3*1)  === targetId && !document.querySelector(`[square-id="${startId-(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2) }"]`).firstChild ||
            startId -  (4*1)  === targetId && !document.querySelector(`[square-id="${startId-(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(3) }"]`).firstChild ||
            startId -  (5*1)  === targetId && !document.querySelector(`[square-id="${startId-(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(3) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(4) }"]`).firstChild ||
            startId -  (6 *1) === targetId && !document.querySelector(`[square-id="${startId-(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(3) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(4) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(5) }"]`).firstChild ||
            startId -  (7 *1) === targetId && !document.querySelector(`[square-id="${startId-(1) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(2) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(3) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(4) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(5) }"]`).firstChild && !document.querySelector(`[square-id="${startId-(6) }"]`).firstChild 
        ){
            return true;
        }
        break;

        case 'king':
            if(
                startId +1 === targetId ||
                startId -1 === targetId ||
                startId +width === targetId ||
                startId -width === targetId ||
                startId +width +1 === targetId ||
                startId +width -1 === targetId ||
                startId -width -1 === targetId ||
                startId -width +1 === targetId
                
            ){
                return true;
            }
    }
}


function changePlayer(){
    // here we are just changing the colors of the which is supposed to be moved and we are reversing the id's of the pieces so we dont have the rewrite code for both ways 
    if(playerGo === 'black'){

        reverseIds(); // reverse when player is white 
        playerGo = "white"; // change playergo to white 
        playerDisplay.textContent =playerGo.toUpperCase();
    }
    else{
        revertIds(); //this is to call the revert function when black is the player 
        playerGo = "black";  //vice-versa
        playerDisplay.textContent =playerGo.toUpperCase();
    }
}
//reverse id's
function reverseIds(){
    // calculation for reversal of ids of the squares 
    const allSquares = document.querySelectorAll(".square");
    
    allSquares.forEach((square, i) => // FOR EACH QUARE WE GET THE INDEX
        square.setAttribute('square-id', (width * width - 1) - i)) ; //reverse of the indexes of the square by setting the attribute to 63-i where i = 0 for the first index and so on and so forth 

}
function revertIds(){

    // we are making the ids back to normal , this is when blacks turn
    const allSquares = document.querySelectorAll(".square");
    allSquares.forEach((square, i) => // FOR EACH QUARE WE GET THE INDEX
        square.setAttribute('square-id', i)) ; //revert indexes to indexes such as 0,1...

}

function checkforWin(){
    const kings = Array.from(document.querySelectorAll('#king'))
    console.log(kings) ;
    if(!kings.some(king => king.firstChild.classList.contains('black'))){
        infoDisplay.innerHTML = "WHITE PLAYER WINS ! ";
        playerDisplay.textContent = "HAHAHA BLACK LOST ";
        disableDraggables(); 
        
    }

    if(!kings.some(king => king.firstChild.classList.contains('white'))){
        infoDisplay.innerHTML = "BLACK PLAYER WINS ! ";
        playerDisplay.textContent = "OVER";
        disableDraggable();
        disableDraggable();
    }
}

function disableDraggable(){
    const allSquares=document.querySelectorAll('.square');
    allSquares.forEach(square => square.firstChild?.setAttribute('draggable', false))
    allSquares.forEach(square => square.firstChild?.setAttribute('draggable', false))
}



function checknextmoves(){
    //i want to check all the possible moves made by each and every piece which is not covered or somerthing so i want to highlight those moves 
    // i was to check if the that a king can move to a square which is not a possible move from any other oppositions piece move , if it is then we stop the king from moving there 
    // when a piece is taken , the taken piece is added to the tally  which will show which side is up by how many points 

}