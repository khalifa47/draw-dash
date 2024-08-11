"use client";
import * as React from "react";
import Image from "next/image";

import { Dialog, DialogContent, DialogTrigger } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

import { Star } from "lucide-react";

import { Card } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

import { Label } from "@/components/ui/label";
import { useMagicContext } from "../providers/MagicProvider";
import { NFTMarketplaceAbi } from "@/lib/abi/NFTMarketplaceABI";

export default function CreationCard({
  creation,
  account,
}: {
  creation: any;
  account: any;
}) {
  const { web3 } = useMagicContext();

  const MarketplaceAddress = "0xa647c2a9032CAa06f721D79f0c05E1304cbfe0bC";
  const MintAddress = "0x6288541D44Cd7E575711213798dEA5d94417519B";

  async function buyNFT() {
    if (!web3) throw new Error("Web3 not connected");
    const fromAddress = (await web3.eth.getAccounts())[0];

    const contract = new web3.eth.Contract(
      NFTMarketplaceAbi,
      MarketplaceAddress
    );

    const receipt = await contract.methods.buyNFT(MintAddress, 1).send({
      from: fromAddress,
    });
    console.log("receipt", receipt);
  }

  async function listNFT() {
    if (!web3) throw new Error("Web3 not connected");
    const fromAddress = (await web3.eth.getAccounts())[0];

    const contract = new web3.eth.Contract(
      NFTMarketplaceAbi,
      MarketplaceAddress
    );

    const receipt = await contract.methods.listNFT(MintAddress, 1, 1).send({
      from: fromAddress,
    });
    console.log("receipt", receipt);
  }

  return (
    <>
      <Dialog>
        <DialogTrigger asChild>
          <Card>
            <div className="w-full p-1 ">
              <Image
                src={creation.enhanced_image || creation.drawing_url}
                width={200}
                height={200}
                alt={"game"}
                className="aspect-[1] rounded-md border w-full"
              />
            </div>

            <div className="flex  m-2">
              <Avatar className="w-8 h-8">
                <AvatarImage src={creation.owner?.picture} alt="@shadcn" />
                <AvatarFallback>CN</AvatarFallback>
              </Avatar>
              <div className="flex flex-col ml-2 mt-1">
                <Label className="font-semibold">{creation.name}</Label>
                <Label className="text-xs text-gray-500">
                  {" "}
                  by {creation.owner?.name}
                </Label>
              </div>
            </div>
          </Card>
        </DialogTrigger>
        <DialogContent className="max-w-3xl">
          <div className="grid grid-cols-12">
            <div className="col-span-6">
              <div className="border rounded-xl p-1 m-1 ">
                <Image
                  src={creation.enhanced_image || creation.drawing_url}
                  alt={"game"}
                  width={500}
                  height={500}
                  className="aspect-[1] rounded-lg "
                />
              </div>
            </div>
            <div className="col-span-6">
              <div className="flex  m-2 px-2">
                <Avatar className="w-12 h-12">
                  <AvatarImage src={creation.owner?.picture} alt="@shadcn" />
                  <AvatarFallback>CN</AvatarFallback>
                </Avatar>
                <div className="flex flex-col ml-2 ">
                  <Label className="font-semibold text-xl">
                    {creation.name}
                  </Label>
                  <Label className="text-xs text-gray-500">
                    {" "}
                    by {creation.owner?.name}
                  </Label>
                </div>
              </div>
              <div className="flex mx-2">
                <div className="border rounded-xl p-1 m-1 ">
                  <Image
                    src={creation.drawing_url}
                    alt={"game"}
                    width={140}
                    height={140}
                    className="aspect-[1] rounded-xl "
                  />
                </div>
                <div>
                  <div className="text-sm font-semibold px-2">prompt</div>
                  <div className="border rounded-xl px-2 py-2 m-1 text-xs bg-indigo-50">
                    {creation.prompt}
                  </div>
                </div>
              </div>
              <div className="border rounded-xl px-4 pt-2  pb-4 m-3 bg-gray-50">
                <p className="flex font-semibold  px-2 mb-0.5">
                  Creation details
                </p>
                <p className="flex text-xs  px-2 mb-0.5 font-semibold">
                  Score: {creation.score}
                  <Star className="w-3.5 h-3.5 ml-1 text-amber-500" />
                </p>
                <p className="text-xs  px-2 mb-0.5">
                  Similarity: {creation.similarity}
                </p>
                <p className="text-xs  px-2 mb-0.5">
                  Time taken: {creation.time_taken} secs
                </p>
                <p className="text-xs  px-2 mb-0.5 font-semibold">
                  Price: {creation.price} usdc
                </p>
              </div>
              {creation.listed && creation.owner?._id !== account?._id && (
                <Button
                  className="w-full mt-2 px-6 py-3 mx-2 "
                  onClick={buyNFT}
                >
                  Buy NFT
                </Button>
              )}
              {!creation.listed &&
                creation.minted &&
                creation.owner?._id === account?._id && (
                  <Button
                    className="w-full mt-2 px-6 py-3 mx-2 "
                    onClick={listNFT}
                  >
                    List NFT
                  </Button>
                )}
              {!creation.minted && creation.owner?._id === account?._id && (
                <Button className="w-full mt-2 px-6 py-3 mx-2 ">
                  Mint creation as NFT
                </Button>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
