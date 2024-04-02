function* fetchnextelement(){
    const x=10;
    yield 11;
    console.log("entering after a yield");

    const y=x+ (yield 30);
    console.log("value of y is",y);
}

const iter=fetchnextelement();

console.log("first",iter.next());
console.log("second",iter.next());
console.log("third",iter.next(10));