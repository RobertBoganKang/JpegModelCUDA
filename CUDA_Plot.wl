(* ::Package:: *)

dat=ToExpression@First@First@Import@(NotebookDirectory[]<>"/a.dat");
Image[dat,"Byte",ImageSize->Large]//ImageAdjust
