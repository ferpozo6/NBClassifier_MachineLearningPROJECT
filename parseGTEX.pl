#!/usr/bin/perl

use strict;

my %hash;
open FILE, "gtex_Brain_phenotype.txt";
while (<FILE>){
	chomp;
	my @arr = split ' ';
	$hash{$arr[1]}{$arr[0]}->{'line'} = $_;
	$hash{$arr[1]}{$arr[0]}->{'gender'} = $arr[4];
	#print "$arr[0] $arr[1] $arr[2] ... \n"; 
}
close FILE;

my $countMale = 0;
my $countFemale = 0;
foreach my $tissue (sort keys %hash){
	foreach my $id (sort keys %{$hash{$tissue}}){
		my $gender = $hash{$tissue}->{$id}->{'gender'};

		if ($gender eq 'Male' and $countMale <= 29){ 
			#print "$tissue $id ($gender):\n\t$hash{$tissue}->{$id}->{'line'}\n";
			print "$hash{$tissue}->{$id}->{'line'}\n";
			$countMale++;
		}
		if ($gender eq 'Female' and $countFemale <= 14){
			#print "$tissue $id ($gender):\n\t$hash{$tissue}->{$id}->{'line'}\n";
			print "$hash{$tissue}->{$id}->{'line'}\n";
			$countFemale++;
		}
		
	}
	$countMale = 0;
	$countFemale = 0;
}

 ##### CHOOSE THE NUMBER OF MALES AND FEMALES THAT YOU WANT FOR YOUR TRAINING SET !!
